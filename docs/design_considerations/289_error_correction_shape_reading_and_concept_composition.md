# Design Consideration 289: Error Correction, Shape Reading, and Concept Composition

## Status: Theoretical Synthesis
## Date: 2026-03-04
## Prerequisites: DC 258, DC 288, F157, Base64_BBP

## Summary

Three insights converge from the F157 weight structure analysis, DC 258's
φ-π Rosetta Stone, and the Base64_BBP formula:

1. The gate anti-alternation across layers IS Newtonian error correction
2. The shapes created by weights can be read directly (geometry head)
3. Concepts compose geometrically as relative positions in shape space

Together these suggest: **the transformer is a convergent error-correcting
encoder whose output shapes ARE readable geometric concepts that compose
relationally.**

---

## 1. Alternating Layers as Newtonian Error Correction

### 1.1 The Leibniz Mechanism

The Leibniz-Gregory series converges to π/4 through alternating signs:

```
π/4 = 1 - 1/3 + 1/5 - 1/7 + ...
```

Each term overshoots, the next corrects. After N terms, the error is
bounded by |1/(2N+1)|. The key insight from DC 258 §2.2:

> "Alternation is the mechanism, not just a mathematical convenience."

### 1.2 The Gate Anti-Alternation Pattern

F157 discovered that gate_proj singular vectors anti-correlate across
adjacent layers: cosine similarity flips sign layer-to-layer, with
|cos| ≈ 0.8. This IS the Leibniz mechanism operating in weight space:

```
Layer n:   gate pushes in direction +d
Layer n+1: gate pushes in direction -d (correcting the overshoot)
Layer n+2: gate pushes in direction +d (correcting the correction)
...
After 28 layers: cumulative error converges
```

Critically, |cos| ≈ 0.8 (not 1.0). Each correction is slightly LESS
than the overshoot — which is exactly what convergence requires. If
corrections were equal, you'd oscillate forever. The 0.8 factor is the
damping ratio that ensures convergence.

DC 258 predicted the convergence accuracy: 1/(4φ⁴) ≈ 3.6%. The
sequential residual was measured at 3.61%. This is the Leibniz error
bound applied to 28 layers of alternating gate injection.

### 1.3 Error Correction Implies Encoding

This is the key insight: **if the alternating pattern IS error
correction, then the structure is ENCODING information through the
convergence process.**

The Leibniz series encodes π/4. Each term is not "computing" π/4 from
scratch — it's refining a representation that converges to π/4. The
alternating gate layers encode the target representation the same way:
each layer refines the previous layer's output through push-pull
correction.

This reframes the 28-layer transformer: it's not 28 independent
computations chained together. It's a single convergent encoding
process where each layer is a correction term in an alternating series.

### 1.4 Connection to Base64_BBP

The Base64_BBP dual alternating series:

```
π/4 = (1/16) Σ (-1)^n/64^n [8/(4n+1) + 4/(4n+2) + 1/(4n+3)]
    + (1/256) Σ (-1)^n/1024^n [32/(4n+1) + 8/(4n+2) + 1/(4n+3)]
```

Features:
- **(-1)^n alternation** — same as gate anti-alternation
- **Dual series** — same as gate ⊥ up independent channels
- **4-periodic denominators** — same as 4-state gate dimension
- **Rapid convergence** — ~100 terms for 100+ digits

The transformer's alternating gate architecture mirrors this:
two independent channels (gate and up), alternating corrections,
converging to the answer.

---

## 2. arctan(1/φ) + arctan(1/φ³) = π/4 as Push-Pull Geometry

### 2.1 The Formula Reads as Two Complementary Forces

```
arctan(1/φ) + arctan(1/φ³) = π/4     (EXACT)
```

- arctan(1/φ) ≈ 31.7° — the dominant "push"
- arctan(1/φ³) ≈ 13.3° — the corrective "pull"
- Together: exactly 45° = π/4

This is NOT just an identity. It's the decomposition of a quarter-turn
rotation into two sub-rotations scaled by φ.

### 2.2 Push-Pull in 4 Dimensions

In 3D, a rotation has one axis and one angle.
In 4D, a rotation requires TWO planes and TWO angles.

The Fibonacci arctan identity decomposes a π/4 rotation into:
- Rotation by arctan(1/φ) in the φ-plane
- Rotation by arctan(1/φ³) in the φ³-plane

These are two independent rotation planes that together complete
a quarter-turn. This maps directly to:

- **W_gate**: the "push" — selects which features activate
- **W_up**: the "pull" — provides values to inject
- **After W_down**: orthogonal (cos ≈ 0.000) — independent planes
- **Together**: injection into the residual's null-space (π/4
  complementarity in the gate dimension, DC 258 §3.1)

The MLP's two independent channels (F157: gate ⊥ up after compress)
ARE the two rotation planes in the arctan decomposition. The φ-scaling
isn't accidental — it's the natural decomposition of a 4D rotation.

### 2.3 Error Correction as Encoding = Rotation as Information

Combining §1 and §2: the alternating error correction IS a rotation
that converges to a target orientation. Each correction term rotates
the representation slightly, alternating between the φ-plane and
φ³-plane. After 28 layers (= 4φ⁴), the representation has been
rotated to within 1/(4φ⁴) of the target.

The encoding is the rotation itself. The shape of the representation
after convergence IS the encoded information.

---

## 3. Reading Shapes Directly: The Geometry Head

### 3.1 The Current Decode Path

The standard transformer output path:

```
hidden_state → LM head (152K dot products) → logits → softmax → token
```

The LM head IS already a shape-reading operation: 152,064 direction
vectors (one per token), and the hidden state's dot product with each
gives the "shape similarity" to each token.

But this is the BRUTE FORCE version — comparing against every possible
shape. If the geometric structure has regularity (and F157 shows it
does), we should be able to read shapes more efficiently.

### 3.2 The Geometry Head Concept

Instead of comparing against 152K token directions, a geometry head
would read the geometric structure of the hidden state directly:

```
hidden_state → SVD basis projection → shape coordinates → lookup
```

This is analogous to the BBP spigot algorithm: instead of computing
all preceding digits of π to get digit N, BBP extracts digit N
directly from the structure. A geometry head extracts the answer
token directly from the geometric structure of the hidden state.

ShapeSpace already does this in miniature: it projects a query vector
into a geometric subspace and finds the nearest answer. The difference
is ShapeSpace works on extracted fact-type-specific subspaces. A
general geometry head would work on the full output space.

### 3.3 What the Selector Comb Machine Tells Us

The geometric instrument (Exp 5b) showed that Layer 1 operates as a
rank-1 selector bank: 28 heads, each projecting tokens onto a single
direction and selecting the maximum. The selector is reading a shape
— the shape of the hidden state projected onto 28 geometric axes.

The selector comb machine extends this: the comb layers (6-25) each
apply their gate push-pull correction, and the selector reads the
resulting shape. The output IS the shape created by 28 layers of
alternating error correction.

If the selector comb is outputting expected bytecodes, it's because:
1. The weights create shapes via alternating error correction (§1)
2. Those shapes converge to the geometric representation of the answer
3. The selector reads those shapes

The bytecodes ARE the shapes, read in binary. The model's output
layer is a shape-to-binary converter.

### 3.4 Custom Geometry Head Architecture

A custom geometry head would:

1. **Project** the hidden state into the SVD basis of the output space
   (not 152K comparisons, but d-dimensional coordinates)
2. **Read** the shape coordinates as a structured code
   (like BBP reading digits without computing predecessors)
3. **Decode** the geometric code to a token
   (like Base64_BBP: the dual series structure gives the answer)

The key insight: we don't need to compare against ALL tokens.
If the shape space is structured (and F157 shows it is), the
shape coordinates already SPECIFY the token. We just need to
learn to read them.

---

## 4. Concept Composition: "Dragon Shrimp" → "Lobster"

### 4.1 Concepts as Shapes

If the hidden state after 28 layers of error-correcting convergence
IS the geometric representation of a concept, then concepts have
shapes. These shapes live in a d-dimensional space where position
encodes meaning.

"Dragon" has a shape. "Shrimp" has a shape. What happens when you
compose them?

### 4.2 The Composition Hypothesis

```
shape(dragon) ⊕ shape(shrimp) ≈ shape(lobster)
```

This is the word2vec analogy (king - man + woman = queen) but richer:
- word2vec uses single vectors (points in space)
- We're working with SHAPES (multi-dimensional geometric structures)
- The composition operation ⊕ could be addition, rotation,
  element-wise product, or something more geometric

Why "lobster"? Because a lobster is:
- Large + armored (dragon properties)
- Aquatic + crustacean (shrimp properties)
- The INTERSECTION of these property sets in concept space

The geometric operation isn't just vector addition — it's finding the
concept whose shape is most consistent with BOTH input shapes. This
is a constraint satisfaction problem in shape space.

### 4.3 Concepts Relative to Other Concepts

The deeper insight: "lobster" isn't defined absolutely. It's defined
by its relationships to other concepts:

- More armored than a shrimp
- More aquatic than a dragon
- Similar size to a cat
- Similar diet to a crab
- ...

The geometric representation IS the set of relative positions. In
ShapeSpace, entity positions ARE their answer-relative coordinates
(this is what whitened alignment achieves — DC 285).

"Dragon shrimp" → "lobster" works because:
1. "Dragon" occupies a position defined by its relationships
2. "Shrimp" occupies a position defined by its relationships
3. Their composition creates a NEW position
4. That position's nearest neighbor in concept space is "lobster"
5. It's "lobster" BECAUSE the relationship pattern matches

### 4.4 The Language of Shapes

If concepts compose geometrically, then there IS a "language of
shapes." Just as written language composes phonemes into morphemes
into words into sentences, shape language composes:

- Feature directions into concept positions
- Concept positions into relational structures
- Relational structures into compositional meanings

The alternating error correction (§1) is the GRAMMAR of this
language: it specifies how shapes converge to valid concepts.
The push-pull φ-structure (§2) is the PHONOLOGY: the fundamental
units of geometric rotation. The geometry head (§3) is the READER:
it converts shapes back to tokens.

---

## 5. The Unified Picture

```
ENCODING (28 layers of alternating error correction)
    │
    │  Each layer: gate pushes, next layer pulls back
    │  Convergence: 1/(4φ⁴) ≈ 3.6% residual error
    │  Push-pull: arctan(1/φ) + arctan(1/φ³) = π/4
    │  Two channels: gate ⊥ up (independent rotation planes)
    │
    ▼
SHAPE (geometric representation in d-dimensional space)
    │
    │  The shape IS the concept
    │  Position encodes meaning relationally
    │  Concepts compose by geometric operations
    │  "Dragon" ⊕ "shrimp" → "lobster"
    │
    ▼
DECODING (reading the shape)
    │
    │  Current: brute-force 152K comparisons (LM head)
    │  Proposed: geometry head reads shape coordinates directly
    │  Analogy: BBP digit extraction without computing predecessors
    │
    ▼
OUTPUT (token / bytecode / concept)
```

The alternating error correction IS the encoding.
The shape IS the information.
The geometry head IS the decoder.
Encoding = decoding in opposite directions (DC philosophy: ENCODE = DECODE).

---

## 6. Testable Predictions

### 6.1 Alternation Convergence Test

**Prediction:** The cumulative gate injection should converge like
an alternating series. Measuring ||residual|| after layers 1, 2, 3...
should show oscillating decrease with bound ~1/layer.

**Method:** Run a forward pass, record the gate_proj contribution at
each layer, plot cumulative sum. Should show alternating overshoot/
correction with decreasing amplitude.

### 6.2 Geometry Head Prototype

**Prediction:** A d-dimensional projection (d << 152K) should suffice
to identify the correct output token for known fact types.

**Method:** For the capital-of task, project the final hidden state
into the SVD basis of answer embeddings. Measure how many dimensions
are needed to uniquely identify the correct capital. ShapeSpace already
achieves this with d ≈ 10-15 dimensions.

### 6.3 Concept Composition Test — F158 Results

**Prediction:** shape(A) + shape(B) should have its nearest neighbor
be semantically consistent with the composition "A-like B."

**Result (F158, raw embeddings):** PARTIAL CONFIRMATION.

- dragon + shrimp → lobster at **rank 17** out of 152K (top 0.01%)
- Paris - France + Germany → Berlin at **rank 3** (geographic analogy)
- foot + ball → football at rank 8
- rain + bow → rainbow at rank 18
- Addition and average work equally; element-wise multiply FAILS

**Relational fingerprint (24 reference concepts):**
- eagle → phoenix (✓ raptor/mythical bird)
- castle → Castle (✓ identity recovered)
- lobster → Dolphin (~ aquatic but wrong taxon)

**Why dragon+shrimp works:** "Dragon shrimp" (龙虾, lóngxiā) IS
literally the Chinese word for lobster. Qwen2-7B has full Chinese
support. The embedding space encodes **conceptual structure that
transcends language** — the shapes represent CONCEPTS, not words.
Chinese makes this structure explicit; English hides it.

**Critical caveat:** These are RAW EMBEDDINGS (layer 0). The real
shapes emerge after 28 layers of error-correcting convergence.
The signal exists but is weak. The next test: compose in the
converged hidden state (post-layer-27) where shapes are fully
developed. If the error correction hypothesis holds, rank should
improve dramatically (17 → <5).

### 6.4 Push-Pull Orthogonality Test

**Prediction:** The gate and up contributions at each layer should be
orthogonal AND their relative magnitudes should relate by φ.

**Method:** During forward pass, record gate_proj and up_proj
contributions separately. Measure cos(gate, up) and ||gate||/||up||
at each layer. If the arctan decomposition holds, the ratio should
involve φ.

---

## 7. Connection to Prior Work

| Prior Work | Prediction | This DC |
|:-----------|:-----------|:--------|
| DC 258 §2.2 | Alternation = convergence mechanism | Gate anti-alternation IS Leibniz convergence |
| DC 258 §3.2 | Two independent chirality channels | Gate ⊥ up = two rotation planes |
| DC 258 §5.4 | BBP = spigot for gate states | Geometry head = spigot for output tokens |
| DC 288 §6.3 | MLP = conditional orthogonal injector | Push-pull = arctan(1/φ) + arctan(1/φ³) |
| DC 285 | ShapeSpace encodes entities relationally | Concepts compose as relative positions |
| Base64_BBP | Dual alternating series converge to π/4 | Dual channels (gate/up) converge to answer |
| F157 | Gate anti-alternation across layers | = error correction terms in alternating series |

---

## 8. Experimental Results (F159)

All three experiments from §4-6 have been run. **All three hypotheses were
disproved in their naive form**, but each reveals structural information.

### 8.1. Experiment 1: NOT Alternation — "Inflate-Process-Deflate"

The MLP contributions do NOT alternate. Only 3/27 consecutive pairs flip
direction (11%). Instead, 99.4% of MLP variance lies along a SINGLE direction,
and the model exhibits a three-phase structure:

- **L0-L17 (Build)**: Cumulative MLP grows from 13 to 9464. L3 alone injects
  ‖MLP‖=7537 — one massive identity vector.
- **L18-L26 (Retract)**: Cumulative shrinks from 9462 to 2995. L26 alone
  removes ‖MLP‖=6157 in the opposite direction.
- **L27 (Final)**: Attention dominates (‖attn‖=3090 vs ‖mlp‖=215).

This is **inflate → process → deflate**: the MLP builds a scaffold, the model
works on it across layers, then the scaffold is removed before output.

**Implication for TruthSpace**: The "error correction" in transformers is not
a simple alternating series. The geometric analog would be: inflate a concept
into a high-magnitude working representation, apply transformations, then
project back down. The convergence we saw in gate anti-alternation (F157) may
operate at a sub-layer level, not across the full layer stack.

### 8.2. Experiment 2: Composition WORKS in Embeddings, FAILS in Hidden States

| Composition | Embed rank | L06 | L13 | L20 | L26 | Final |
|-------------|-----------|-----|-----|-----|-----|-------|
| dragon+shrimp→lobster | **17** | 97K | 97K | 97K | 99K | 55K |
| rain+bow→rainbow | **18** | 126K | 126K | 126K | 124K | 19K |
| foot+ball→football | **8** | 65K | 65K | 65K | 66K | 19K |

**Embedding-level composition is the best.** Deep layers produce garbage
(top tokens are German/Japanese: 'ĠcarÃ¡', 'ãģıãģł', 'Ġerfolgre').

Why: hidden states from single-token passes encode **processing state**, not
concept identity. The L3 scaffold injection puts each token onto its own
massive identity vector. Adding scaffolds produces nonsense.

**Implication for TruthSpace**: The embedding space IS concept space.
Composition should happen there, not after layer processing. The 28-layer
transform is optimized for next-token prediction in context, not for
preserving concept geometry.

### 8.3. Experiment 3: Output Space is Full-Rank

The top predicted token ('ĠBall' for "dragon") requires **2000/3584 SVD
dimensions (55.8%)** to become rank 1. Variance is spread broadly: top 100
dims capture only 15.2%.

This confirms F147: weight matrices are essentially full-rank (Zipf α ≈ 0.12).
There is no low-dimensional geometric shortcut for token identification.

**Implication for TruthSpace**: A geometry head cannot simply be a low-rank
projection of lm_head. Token discrimination genuinely uses the full
dimensionality. Any geometric replacement must find a DIFFERENT basis for
token identification — not SVD compression of the existing one.

### 8.4. Revised Understanding

The real geometric signal is in the **embeddings**, not the layers:
- Rank 8-18 for compound word composition IS good (out of 152K vocabulary)
- The 28-layer pipeline is a processing engine, not a concept refiner
- Concept composition via vector addition works at the embedding level
- The Chinese conceptual structure (龙虾 = dragon+shrimp = lobster) is
  encoded in the embedding geometry by training

---

## 9. Open Questions (Updated)

1. **What IS the composition operator?** Simple addition works at
   the embedding level (rank 8-18 out of 152K). Can we do better with
   weighted addition, or geometric operations like rotation/reflection?
   (F159 showed addition ONLY works in embedding space, not hidden states.)

2. **What is the inflate-process-deflate scaffold?** L3 injects a massive
   identity vector, L26 removes it. What does this scaffold encode? Is it
   the same direction for all tokens, or token-specific? If we remove L3
   and L26, does the model still function on the remaining layers?

3. **Can we build a geometry head from embedding structure?** SVD
   compression of lm_head doesn't work (55.8% dims needed). But the
   embedding space has compositional structure (rank 8-18). Can we build
   a decoder that works in embedding space directly, bypassing lm_head?

4. **Do shapes have a "grammar"?** Are there invalid shapes that
   don't correspond to any concept? What does the manifold of valid
   concept shapes look like in the embedding space?

5. **Is the embedding space already the geometric space we need?**
   F159 showed concepts compose at the embedding level. The 28-layer
   pipeline transforms concepts into processing state. Can TruthSpace
   operate entirely in embedding space, using the layer pipeline only
   when contextual processing is needed?

---

## 10. Files

| File | Purpose |
|:-----|:--------|
| `docs/design_considerations/258_phi_pi_rosetta_stone.md` | DC 258: φ-π structure of gate dimension |
| `docs/design_considerations/288_weight_structure_the_ordering_is_in_the_shape.md` | DC 288: weight structure analysis |
| `experiments/model_reverse_engineering_v2/FINDINGS.md` | F157: gate anti-alternation |
| `experiments/geometric_instrument/shapespace.py` | ShapeSpace: geometric concept encoding |
| Base64_BBP | https://github.com/lostdemeter/Base64_BBP |
