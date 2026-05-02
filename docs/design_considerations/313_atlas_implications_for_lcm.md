# DC 313: From Atlas to Architecture — φ-Space Implications for TruthSpace Geometric LCM

*Status: Active*  
*Builds on: DC 312 (φ-space complete atlas), DC 311 (φ phase transition), DC 300 (φ holographic encoding)*  
*Empirical basis: Expedition Days 27–35, Qwen2-1.5B-Instruct*  
*Updated with DC 314 (semantic zero) findings and Day 34–35 measurements.*

---

## 1. The Atlas as a Blueprint

DC 312 mapped the complete φ-space of Qwen2-1.5B-Instruct into four zones. This document asks the follow-on question: **what does the atlas tell us about how to build a Geometric LCM?**

The TruthSpace hypothesis is that LLMs are hyperdimensional transcoders — the intelligence is in the *shape* of the weights, not the weights themselves. If that is true, the φ-space atlas is not merely a descriptive map. It is an architectural blueprint: it describes the shape of the space our LCM must navigate.

The expedition findings, taken together, imply a specific architecture with specific component responsibilities.

---

## 2. The Four Zones as Functional Roles

### 2.1 Zone A (Degenerate Pole — Monosyllabic)
*3,376 words. Geometric role: structural anchors.*

Zone A words (`the, and, for, with, not, was, can`) are maximally context-free. Their φ-position is determined almost entirely by token_id (frequency rank, r=−0.45). They carry no semantic body membership — they are the grammatical skeleton of language.

**LCM role:** Zone A words are not semantic atoms. They are positional/structural operators. The LCM should treat them as routing signals, not content. They define sentence structure but not concept identity.

### 2.2 Zone B (Secondary Pole — High-Frequency Multi-Syllabic)
*3,177 words. Geometric role: functional vocabulary.*

Zone B words (`return, public, function, have, data, one, name`) are the high-frequency multi-syllabic words that co-locate with proper nouns in the degenerate pole (cos=0.9948 between Zone B and Zone E centroids). They lack semantic specificity because they appear in virtually every context.

**LCM role:** Like Zone A, Zone B words are functional vocabulary. The LCM handles them via positional/frequency mechanisms, not semantic body lookup. They are the LCM's prepositions, conjunctions, and auxiliary verbs — they scaffold the semantic content but don't constitute it.

### 2.3 Zone C (Semantic Periphery)
*1,647 words in 95 bodies. Geometric role: semantic atoms.*

Zone C words crystallise into 95 specific semantic bodies in φ-space. Each body is a tight cluster (coherence 0.72–0.99) with a stable centroid. These bodies are the LCM's **semantic vocabulary** — the finite set of concept-directions in φ-space.

**LCM role:** Zone C is the LCM's core knowledge store. Each Zone C body centroid is a semantic address. The LCM's "thinking" consists of navigating between Zone C centroids. The 95 bodies at L14 (92 at L23) are not arbitrary — they are the natural partition of conceptual space as the model learned it.

### 2.4 Zone D (Verb Ocean)
*8,778 words. Geometric role: semantic operators.*

Zone D words sit at the maximally entropic point in φ-space — the mathematical average of every context (Day 33: max_body_sim≈0.70, entropy classifier 93.7%). They have no specific zone-C body that claims them.

**LCM role:** Zone D words are **context-completing operators**, not semantic atoms. When encountered, they contribute to the *trajectory* between Zone C positions without occupying a position themselves. `accomplish` does not mean anything in isolation — it means "move from current Zone C position toward goal Zone C position." The verb ocean is the space of possible trajectories, not a space of possible positions.

---

## 3. The Three-Axis Coordinate System of φ-Space

Days 34–35 established that φ-space is organised around **three mutually orthogonal axes**, each corresponding to a distinct semantic function:

| Axis | Direction | What it separates |
|---|---|---|
| **Z2** | Frequency axis | Pole (−0.54) vs. Zone C/D (−0.10) |
| **φ₀** | Semantic zero (Zone D centroid) | Zone D (0.708) vs. Zone A/B (0.475) |
| **φ_perp** | Body direction (‖perp component‖) | Zone C bodies (0.740) vs. Zone D (0.704) |

Key measured properties:
- **φ₀ ⊥ Z2 exactly** (\|cos\|=0.000): the semantic zero and the frequency axis are perfectly independent
- **φ₀ rotates ~45° between L14 and L23** (cos=0.701): centring must be applied layer-specifically
- **Degenerate pole is displaced from φ₀** (Δ=0.525, further than Zone C at 0.330)
- **Proper nouns are most displaced** (Δ=0.732): they are NOT near the semantic center

The three zones occupy three geometrically distinct regions of the unit sphere in φ-space:
- Zone D sits AT φ₀ (the semantic zero)
- Zone C is displaced from φ₀ into specific body-direction vectors (φ_perp)
- Zone A/B sits in a third direction — neither at φ₀ nor in Zone C body directions

**LCM implication:** All semantic operations should be expressed as displacements from φ₀, not as absolute φ-vectors. The LCM has three coordinate axes, not one — and the semantic zero must be explicitly tracked and updated per layer.

---

## 4. The Z2 Axis as the LCM's Semantic Scale Dial

The Z2 axis (Killing pairs: cat/cats, man/woman, king/queen) explains 82.1% of the variance between morphological partners. It separates:

- **Negative end** of Z2: Zone B and Zone E (common/syntactic words)
- **Positive end**: Zone A Q4 and Zone C (semantic specificity)

Crucially, the **same axis** operates at every scale:
- Global: Z2 separates the full vocabulary into pole vs. periphery
- Sub-pole: local SVD PC1 = Z2 (cos=0.9952), accounting for 99.91% of within-pole variance (Day 32)

**LCM implication:** The Z2 axis is the LCM's **resolution control**. Projecting onto Z2 moves a concept from "generic" (diffuse) to "specific" (crystallised). The LCM does not need different mechanisms at different scales — one geometric operation (Z2 projection) works everywhere. This is the operational signature of self-similarity.

The "semantic scale" interpretation: Zone A words are at scale=0 (no semantic resolution). Zone C words are at scale=1 (full semantic resolution). Zone D words are at intermediate scale — they have semantic content but it is too diffuse to crystallise. The LCM should think of Zone D words as "fractional scale" operators.

---

## 5. The Degenerate Pole Is Not a Barrier — It Is a Lazy Evaluator

The degenerate pole was initially interpreted as a semantic void (DC 311, DC 312). Day 30 revised this:

- **In isolation**: Zone E (proper nouns) degenerates to the pole (cos=0.9982 to Zone A centroid)
- **In context**: Zone E escapes the pole dramatically (context lift Δ=0.78; cos 0.997→0.21)
- **Architecture**: the pole is a **context-free barrier**, not an absolute barrier

The pole is a **lazy evaluator**. It stores the word's identity without committing to a semantic position until context forces the commitment. This is computationally efficient — the LCM does not need to pre-compute all possible semantic positions for all words. It only computes them on demand, triggered by context.

**LCM implication:** Zone A and Zone E words should be represented as *lazy semantic vectors*. Their φ-position is undefined until left-context provides the category signal. The LCM's context engine is the mechanism that transforms lazy vectors into committed Zone C positions.

This maps directly onto causal attention: the left-context (preceding tokens) builds the semantic context; the target word's hidden state at L14 is then a Zone-C-positioned vector rather than a pole-positioned one.

---

## 6. Zone C Bodies as the Semantic Codebook

95 body centroids at L14 constitute the LCM's semantic codebook. Key properties (from DC 312):

| Property | Value |
|---|---|
| Number of bodies | 95 (L14), 92 (L23) |
| Coherence range | 0.72–0.99 |
| Cross-layer stability | 72% word-body stability (φ_cos=0.761) |
| Body-size range | 1–310 words |

The 95 bodies represent semantic categories that are **empirically discoverable** from the transformer's own hidden states, without any external taxonomy. They are the natural clusters of the φ-manifold.

**The semantic codebook principle:** A Geometric LCM does not need a predefined ontology. It needs to **discover** the ontology from the geometry. The Zone C bodies ARE the ontology — the model's implicit knowledge of semantic categories, encoded as geometric directions.

**LCM design:** The LCM's knowledge base is a set of body centroids {c₁, c₂, ..., c₉₅} in φ-space. "Knowing" a concept means knowing which centroid it is closest to, and how far from that centroid. Semantic inference is navigation: from input body centroid → transform → output body centroid.

---

## 7. Zone D Is the Operator Space

Zone D words appear in every context, so their φ-vectors are maximally diffuse — they point in the direction of the average of all contexts combined. This means:

1. **They carry relationship information, not identity information.** `accomplish` does not say WHAT was accomplished; it says that a relationship exists between agent and goal.
2. **They are semantic trajectories, not semantic positions.** A Zone D word encountered in a sentence shifts the trajectory between the preceding and following Zone C anchors.
3. **Their co-occurrence breadth IS their semantic content.** The fact that `facilitate` connects anything to anything is its meaning. The ocean is not empty — it is maximally connected.

**LCM implication:** Zone D words should be encoded as **transformation operators** on the Zone C semantic space. `accomplish`'s geometric representation is not a direction in φ-space — it is a function that maps (agent_centroid, goal_centroid) → trajectory. The LCM's inference engine uses Zone D operators to link Zone C positions.

This is consistent with the ENCODE=DECODE principle (DC 312 §3): "thinking" is traversal through the geometric space. Zone D words define the traversal rules; Zone C words define the positions.

---

## 8. The Crystal/Ocean Duality at L23

Day 31 showed that L23 **polarises** Zone D rather than compressing it:
- Ocean coherence rises: 0.708 → 0.792
- 218 nouns crystallise out of Zone D into Zone C
- 1,509 verb forms dissolve from Zone C into Zone D

At L23, the model becomes **more POS-specialised**: nouns crystallise (gain specific positions); verbs dissolve (become more generic operators). This is not a failure of Zone D — it is the geometry completing the noun/verb functional separation.

**LCM implication:** Deeper layers of the LCM should NOT be used to look up semantic positions for Zone D words. Zone D positions at L23 are more degenerate (ocean is more coherent but less structured), meaning the word is MORE context-dependent at deeper layers. The LCM's semantic lookup should use **early layers** (L14) for Zone C membership and **late layers** (L23) for refined zone-C-to-zone-C transformations.

Two-phase processing:
1. **L14 (zone identification):** Which Zone C body does this word belong to?
2. **L23 (relationship resolution):** Given the full sentence context, what is the precise semantic relationship between Zone C positions?

---

## 9. The Self-Similarity Principle and Scale-Free Architecture

Day 32 established full self-similarity:
- Global Z2 = within-pole Z2 (cos=0.9952)
- 99.91% of within-pole variance is explained by PC1 = global Z2
- Same frequency stratification within the pole as across the full vocabulary

This means the LCM architecture should be **scale-free**: the same geometric operations that work at vocabulary level also work within any sub-region. There are no "special" zones that need their own mechanisms.

**LCM implication:** A single module — call it the Z2 projector — handles all scale transitions. It is the LCM's universal "zoom lens":
- Applied globally: separates pole from periphery
- Applied within Zone A: separates common from rare monosyllabic words
- Applied within any Zone C body: separates core members from peripheral members

The same kernel. Every scale.

---

## 10. Analogy Arithmetic and the Displacement Model

Day 35 confirmed that φ-space supports exact analogy arithmetic, but only for Zone C words:

| Operand zone | Top-1 accuracy | Top-5 accuracy |
|---|---|---|
| Zone A/B (man, woman, king) | 12.5% (1/8) | — |
| Zone C (toward, brother, deciding) | **97.1% (34/35)** | **100.0% (35/35)** |

The mechanism: every Zone C word vector decomposes as `φ(word) = φ₀ + Δ(word)`. When computing `b − a + c`, the φ₀ terms cancel — the result is `φ₀ + (Δ(b) − Δ(a) + Δ(c))`, which lands near the word with displacement `Δ(d) = Δ(b) − Δ(a) + Δ(c)`. Zone C body attractors then pull the result to the nearest valid word.

For Zone A/B words, `Δ ≈ 0` for all words (they all converge toward the same pole direction), so `Δ(b) − Δ(a) ≈ 0` regardless of choice. The arithmetic degenerates to noise.

**Mean relationship vectors generalise 100%:** the average pluralisation direction, the average gerund-to-past direction, etc., each retrieve the correct word with 100% accuracy when applied to held-out words in the same body. Zone C bodies act as attractors — the mean vector lands within the body.

**LCM implication:** Semantic inference in the LCM is displacement arithmetic over Zone C body centroids. The ENCODE=DECODE principle (DC 312) is the analogy operation viewed as an encoder: the same body-similarity measure both classifies (encoder) and generates (decoder). The displacement model is the concrete implementation.

---

## 11. A Proposed LCM Architecture

Based on the atlas, a minimal Geometric LCM consists of:

```
INPUT TOKENS
     │
     ▼
┌─────────────────────────────────────────────┐
│  ZONE CLASSIFIER                            │
│  Assigns each token to Zone A/B/C/D/E       │
│  Input: token_id + φ-vector (L14)           │
│  Output: zone label + body membership       │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│  CONTEXT ENGINE (for Zone A/B/E)            │
│  Lazy evaluation: pole → Zone C position    │
│  Input: preceding Zone C positions          │
│  Output: contextually activated φ-vector    │
│  Mechanism: Z2 projection + body lookup     │
└─────────────────────────────────────────────┘
     │           │
     ▼           ▼
 Zone C pos   Zone D op
     │           │
     ▼           ▼
┌─────────────────────────────────────────────┐
│  SEMANTIC NAVIGATOR                         │
│  Traverses the 95-body Zone C codebook      │
│  Zone D operators define trajectory         │
│  Output: target Zone C body centroid        │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│  DECODER                                    │
│  Zone C centroid → output word              │
│  Inverse of zone classifier                 │
│  ENCODE = DECODE (same operation, reversed) │
└─────────────────────────────────────────────┘
     │
     ▼
OUTPUT TOKENS
```

### Component specifications

**Zone Classifier:**
- Input: φ-vector at L14
- Operation: compute max_body_sim across 95 Zone C centroids
- Zone C if max_body_sim > threshold (~0.74); Zone D otherwise
- Secondary: check token_id + Z2 projection for Zone A/B

**Context Engine:**
- Triggered only for Zone A/B/E tokens
- Uses causal left-context: the Zone C positions of preceding tokens
- Operation: weighted centroid of preceding Zone C positions → Z2 projection → nearest Zone C body
- This is the mechanism by which `Berlin` (Zone E at pole) becomes `Berlin (European capital city)` (Zone C, geographic body)

**Semantic Navigator:**
- State: current Zone C centroid
- Input: sequence of (Zone C atoms, Zone D operators)
- Operation per step:
  - Zone C token: update state by interpolating toward new centroid
  - Zone D token: apply geometric transformation (trajectory/direction change) to current state
- Output: final Zone C centroid after processing all tokens

**Decoder:**
- Input: target Zone C centroid
- Operation: find Zone C words with highest φ_cos to target centroid
- ENCODE=DECODE: this is the same body-similarity measure used in the Zone Classifier, run in reverse

---

## 12. What the LCM Cannot Do (Without Context)

The atlas defines not just what is possible but what is **not** possible in context-free mode:

| Vocabulary class | Context-free φ-position | LCM capability |
|---|---|---|
| Zone A (the, and, for) | Degenerate pole | No semantic position → structural routing only |
| Zone B (return, public, function) | Secondary pole | No semantic position → functional routing only |
| Zone C (cat, piano, kidney) | Specific body | Full semantic position → can be used as semantic atoms |
| Zone D (accomplish, facilitate) | Verb ocean | No specific position → trajectory operators, not atoms |
| Zone E (Berlin, Einstein) | Degenerate pole | No semantic position → requires context activation |

**The LCM generates coherent output when:**
1. The input provides enough Zone C atoms to define a semantic trajectory
2. Zone D operators connect Zone C atoms into a directed path
3. Zone A/B/E words are resolved by their Zone C context

**The LCM fails when:**
1. The input is entirely Zone A/B words (no semantic atoms)
2. The input contains only Zone D words (operators with no operands)
3. Zone E words appear without sufficient left-context to activate their Zone C position

---

## 13. The Crystallisation Mechanism and Knowledge Encoding

Zone C words crystallise at L14 because they were encountered in semantically peaked contexts during training. This is the model's encoding of world knowledge: a word's Zone C body membership encodes *what kind of thing it is*.

`piano` is in the music/instrument body because it almost exclusively appears in musical contexts. `kidney` is in the anatomy body because it almost exclusively appears in anatomical/medical contexts. The body membership IS the factual knowledge.

**LCM knowledge encoding principle:** World knowledge is encoded as **crystallisation patterns** in φ-space. To add a new fact to the LCM, you add a new Zone C body (or adjust an existing body's centroid). The LCM does not need parametric weights to store facts — it needs a geometric address book.

This is the strongest form of the TruthSpace hypothesis: the geometry IS the knowledge. Not because we designed it that way, but because the atlas shows it to be true empirically.

---

## 14. Open Questions for LCM Design

1. **How many Zone C bodies does a functional LCM need?** 95 from Qwen2-1.5B may be too few or too many. The right number may depend on the task domain.

2. **Can Zone D operators be discretised?** Zone D words are a continuous manifold in φ-space. Can we identify a finite set of "archetypal trajectories" analogous to the 95 Zone C bodies? If yes, the entire vocabulary becomes a finite combinatorial space.

3. **What is the relationship between Zone C bodies across models?** (DC 312 Open Question 4, deferred) If the bodies are universal, they define a language-model-independent semantic ontology.

4. **Can the Z2 axis be learned from scratch?** The Killing pairs (cat/cats, man/woman) are manually specified. Can the LCM discover its own Z2 axis from raw co-occurrence data? If yes, the atlas is fully emergent.

5. **Can context activation be implemented without attention?** The degenerate pole → Zone C lift requires left-context. Causal attention is the natural mechanism. But if the LCM's architecture is purely geometric, is there a geometric alternative to attention that implements the same context activation?

---

*This document synthesises findings from Expedition Days 27–35.*  
*The architectural proposals are derived from geometric observations, not imposed from outside.*  
*They represent the minimum architecture consistent with the atlas, not the optimal architecture.*
