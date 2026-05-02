# Doc 287: Converting the Full Model — From 188 Facts to Everything

**Date:** March 4, 2026
**Status:** Analysis — Scoping the path from proof-of-concept to full conversion
**Prerequisites:** DC 285 (ShapeSpace), DC 286 (Geometric Engine), F155–F156

---

## 0. The Question

We converted 188 facts from Qwen2-7B into a geometric engine that runs
without the model. 100% accuracy. 320,000× fewer operations. 5.3 MB
instead of 29 GB.

**What would it take to convert the entire model?**

This document analyzes three progressively ambitious interpretations
of "convert the entire model," from the straightforward to the
speculative. Each section estimates the concrete engineering effort,
computational cost, storage requirements, and open research questions.

---

## 1. The Model We're Converting

Qwen2-7B architecture:

| Component | Value |
|:----------|:------|
| Hidden dimension | 3,584 |
| Layers | 28 |
| Q attention heads | 28 (head_dim = 128) |
| KV attention heads | 4 (GQA ratio 7:1) |
| MLP intermediate dim | 18,944 |
| Vocabulary | 152,064 tokens |
| Total parameters | 7.3 billion |
| Storage (float32) | 29.2 GB |
| Storage (bfloat16) | 14.6 GB |

Per-layer breakdown:
- **Attention**: 18.4M params (8%) — the reader
- **MLP**: 203.7M params (92%) — the amplifier
- The model is overwhelmingly MLP. Converting the MLP is the main challenge.

---

## 2. Interpretation 1: Extract All Factual Knowledge

### 2.1 What This Means

Scale the current extraction pipeline to capture everything the model
"knows" as entity-relationship-answer triples, stored in ShapeSpaces.
The model is still required for extraction but never needed at query time.

This is our current approach, scaled up.

### 2.2 The Entity Problem

We manually curated 47 countries. The model knows about vastly more:
- Named entities (people, places, organizations): ~50,000–200,000
- Common nouns with factual associations: ~10,000–50,000
- Technical terms, dates, quantities: unknown

**Entity discovery** is itself a research problem. Approaches:
1. **Vocabulary mining**: Scan the 152K vocabulary for entity-like
   tokens. Many entities are single tokens ("France", "Einstein").
2. **Prompt probing**: "Tell me about X" for candidate entities,
   check if the model generates coherent factual responses.
3. **Embedding clustering**: Cluster token embeddings to find
   semantically coherent entity groups.
4. **Knowledge graph alignment**: Cross-reference with Wikidata
   (~100M entities, but the model knows a small fraction).

Realistic estimate: **10,000–50,000 extractable entities** where the
model has reliable factual knowledge.

### 2.3 The Relationship Problem

We defined 4 fact types (capital, language, continent, currency) with
manually crafted prompt templates. The model handles hundreds:
- Geographic (capital, continent, population, area, borders)
- Linguistic (language, script, language family)
- Political (leader, government type, independence year)
- Cultural (religion, cuisine, famous landmarks)
- Biographical (birth year, nationality, profession, works)
- Scientific (chemical formula, atomic number, taxonomy)
- And many more...

**Relationship discovery** approaches:
1. **Template enumeration**: "The {relation} of {entity} is" for
   a curated list of ~50–200 common relations.
2. **Free-form probing**: Ask the model to list facts about entities,
   then extract the relationship patterns.
3. **Knowledge graph alignment**: Map Wikidata predicates to
   prompt templates.

Realistic estimate: **50–200 reliably extractable relationship types**.

### 2.4 Cost Estimates

| Scale | Facts | CPU Time | GPU Time† | Raw Storage | ShapeSpace |
|:------|:------|:---------|:----------|:------------|:-----------|
| Current | 188 | 70 min | — | 7.2 MB | 5.8 MB |
| Phase 1 | 20K | 5 days | 2 hours | 0.6 GB | 0.2 GB |
| Phase 2 | 500K | 127 days | 2.5 days | 14.3 GB | 6 GB |
| Phase 3 | 5M | 3.5 years | 25 days | 143 GB | 60 GB |

†GPU estimate assumes 50× speedup from batched forward passes on a
single A100. Actual speedup depends on batch size and sequence length.

### 2.5 The Dimensionality Question

With whitened alignment, each fact type's ShapeSpace needs d = N-1
dimensions to perfectly separate N entities. For N = 10,000 entities
in a single fact type, d = 9,999. Each ShapeSpace would be:

```
basis:     9,999 × 3,584 × 8 bytes = 287 MB
entities:  10,000 × 9,999 × 8 bytes = 800 MB
answers:   10,000 × 9,999 × 8 bytes = 800 MB
Total per fact type: ~1.9 GB
```

At 50 fact types: **~95 GB**. This is large but fits in RAM.

**Open question**: Does d actually need to be N-1 at large N?
At 47 entities, centered answer directions are all linearly
independent (verified experimentally). At 10,000 entities, many
answers are shared (e.g., thousands of entities have continent =
"Europe"). Shared answers reduce effective dimensionality. A fact
type with only 6 unique answers (continent) needs d ≈ 5, regardless
of N. Capital cities are mostly unique, so d ≈ N-1.

Practical estimate: **d ranges from 5 (continent) to ~N-1 (capital)**
depending on answer diversity. Average d might be ~100–500, not ~N.

### 2.6 Feasibility Assessment

**Phase 1 (1K entities × 20 types = 20K facts)**: ✅ Feasible now.
  - 2 hours on GPU, 0.2 GB storage
  - Requires: entity list curation, template writing, GPU access
  - Engineering effort: ~1 week

**Phase 2 (10K entities × 50 types = 500K facts)**: ✅ Feasible with engineering.
  - 2.5 days on GPU, 6 GB storage
  - Requires: automated entity discovery, batched extraction pipeline
  - Engineering effort: ~2–4 weeks

**Phase 3 (50K entities × 100 types = 5M facts)**: ⚠️ Ambitious.
  - 25 days on GPU, 60 GB storage
  - Requires: relationship discovery, quality validation pipeline
  - Engineering effort: ~2–3 months

### 2.7 Limitations

This approach captures the model's **factual recall** — the lookup
table encoded in its geometry. It does NOT capture:
- Multi-step reasoning
- Text generation quality
- Grammar, style, pragmatics
- Novel compositions
- Anything that requires actual computation beyond lookup

---

## 3. Interpretation 2: Decompose the Weight Matrices

### 3.1 What This Means

Instead of extracting facts through forward passes, decompose the
weight matrices themselves into geometric form. The model's parameters
ARE the geometry — read it directly.

This is more aligned with our hypothesis: if structure IS information,
we should be able to read the structure from the weights without
running the computation.

### 3.2 SVD Decomposition

Every weight matrix W (m × n) = U S V^T. At rank k:
- W_k = U[:,:k] @ diag(S[:k]) @ V[:k,:] ≈ W
- Storage: k(m + n + 1) instead of mn
- Each singular component σᵢ · uᵢ ⊗ vᵢ is a **shape** (DC 284)

Compression ratios at various ranks:

| Matrix | Full Size | Rank-64 | Rank-128 | Rank-256 |
|:-------|:----------|:--------|:---------|:---------|
| W_q (3584²) | 51.4 MB | 1.8 MB (28×) | 3.7 MB (14×) | 7.3 MB (7×) |
| W_gate (18944×3584) | 271.6 MB | 5.8 MB (47×) | 11.5 MB (24×) | 23.1 MB (12×) |
| Embeddings (152K×3584) | 2.18 GB | 39.8 MB (55×) | 79.7 MB (27×) | 159.4 MB (14×) |

Full model at rank-128:
- 28 layers × ~80 MB = 2.2 GB (vs 24.9 GB full = **11× compression**)
- Embeddings + LM head: ~160 MB (vs 4.4 GB full = **27× compression**)
- Total: **~2.4 GB** (vs 29.2 GB full = **12× compression**)

### 3.3 The Effective Rank Question — ANSWERED (F157)

**F157 tested this directly.** The answer: matrices are **nearly
full-rank**.

| Weight Type | Shape | r50 | r90 | r99 |
|:------------|:------|:----|:----|:----|
| q_proj | 3584×3584 | 126 | 397 | 489 |
| k_proj | 512×3584 | 96 | 289 | 440 |
| v_proj | 512×3584 | 161 | 391 | 483 |
| gate_proj | 18944×3584 | 167 | 424 | 492 |
| up_proj | 18944×3584 | 209 | 437 | 493 |
| down_proj | 3584×18944 | 209 | 437 | 494 |

90% of the variance requires ~400 out of 500 singular values.
This confirms DC 243 Part 7: "the full spectral structure matters,
not just the dominant modes."

Individual tasks (like capital-of) use only d=4–46, but the model
encodes thousands of tasks simultaneously and their collective
requirements fill the spectrum. There is no clean low-rank
shortcut for compression without quality loss.

The remaining path: **rank-k truncation with quality measurement.**
The question is not "what rank captures 99% variance?" (answer:
~490) but "what rank preserves output quality?" — which may be
lower if some singular components are redundant across tasks

### 3.4 What the SVD Tells Us

The SVD of the weight matrices isn't just compression — it's
**structural analysis**. Each singular component is a shape
(DC 284), and the singular values tell us how important each
shape is to the model's computation.

For each weight matrix:
- **Top singular components**: high-frequency, universally-used
  shapes (e.g., "this is an entity" direction)
- **Middle singular components**: task-specific shapes
  (e.g., "capital-of" relationship direction)
- **Bottom singular components**: noise or rare-case shapes

The distribution of singular values reveals the model's
**information hierarchy** — how it allocates representational
capacity across different types of knowledge.

### 3.5 Beyond Individual Matrices

The model's computation isn't just individual weight matrices — it's
their composition across layers. The residual stream creates an
effective matrix that's the product of all layer contributions:

```
h_final = h₀ + Σᵢ Δhᵢ(h₀ + Σⱼ<ᵢ Δhⱼ(...))
```

This nested structure means the model's effective computation is
higher-rank than any individual matrix. But the residual stream also
means each layer contributes an additive correction, and many of
these corrections are low-rank.

### 3.6 Feasibility Assessment

**SVD analysis of all weight matrices**: ✅ Straightforward.
  - Already have `UnwoundQwen2` that extracts all weights
  - SVD each matrix, analyze singular value distributions
  - Engineering effort: ~1 day

**Rank-k quality assessment**: ✅ Feasible.
  - Replace each matrix with rank-k approximation
  - Run inference on standard benchmarks
  - Map quality vs rank → find minimum rank per matrix
  - Engineering effort: ~1 week

**Geometric structure discovery**: ⚠️ Research.
  - Identify which singular components correspond to which tasks
  - Map the model's "shape vocabulary"
  - Engineering effort: unknown (research frontier)

---

## 4. Interpretation 3: Full Geometric Computation

### 4.1 What This Means

Replace the entire forward pass with geometric operations. No matrix
multiplies. No softmax. No SiLU. Just direction lookups and traversal
through geometric space.

This is the ultimate validation of the hypothesis: if the intelligence
is in the shape, we should be able to traverse the shape directly.

### 4.2 What's Already Geometric

Much of the forward pass is already geometric in nature:

| Operation | Geometric? | Current Status |
|:----------|:-----------|:---------------|
| Embedding lookup | ✅ Yes | Done (token → direction) |
| Q/K/V projection | ✅ Yes | Linear rotation/projection |
| Attention scores | ✅ Yes | Dot products (geometric similarity) |
| Value readout | ✅ Yes | Similarity-weighted sum |
| Output projection | ✅ Yes | Linear rotation |
| Residual addition | ✅ Yes | Superposition of shapes |
| Layer norm | ⚠️ Mostly | Projection onto sphere |
| Softmax | ❌ No | Nonlinear normalization |
| SiLU gate | ⚠️ Characterized | DC 243: x·σ(φ·x) ≈ GELU; phase transition at α≈1.5 |
| MLP gating | ⚠️ Characterized | DC 243: conditional orthogonal injector |

The primary nonlinear barrier is **softmax**. The SiLU gate is
better understood than previously thought (see §4.4).

### 4.3 The Softmax Barrier

Softmax converts raw attention scores into a probability distribution.
Its effect: normalize so weights sum to 1, and sharpen toward the
maximum (exponential amplification).

Our whitened alignment is the geometric analog: it sharpens the
entity-answer mapping to one-hot without softmax. But whitened
alignment requires knowing all entities in advance (stored lookup).
For novel sequences, we'd need a geometric alternative to softmax.

Candidates:
- **Normalized dot product** (cosine similarity): preserves
  geometric structure but doesn't amplify peaks
- **Power normalization** (scores^p / sum): sharper than cosine,
  still geometric
- **Temperature-scaled projection**: project onto simplex

### 4.4 The MLP — Not the Barrier We Thought

The MLP computes:

```
output = SiLU(h @ W_gate.T) ⊙ (h @ W_up.T) @ W_down.T
```

We initially called this "the hard barrier." Three prior results
and one new experiment show it's better characterized than that:

**DC 243 (GELU Machine):** SiLU is x·σ(1.0·x), which is
catastrophically broken (α=1.0 is below the phase transition at
α≈1.5). GELU ≈ x·σ(φ·x) works because φ is the critical transition
constant. The MLP is a "conditional orthogonal injector" — W_compress
lives 65% in null(W_expand), injecting new information rather than
reconstructing. The SV spectrum is the information, not the
directions. When the spectral structure is φ-correct, the gate
choice becomes irrelevant.

**F157 (Weight Structure):** Confirms DC 243 for Qwen2-7B:
- gate_proj ⊥ k_proj (cos=-0.524) — gate looks OPPOSITE to key
- Gate direction anti-alternates across layers (|cos|≈0.8)
- The anti-alternation IS the orthogonal injection pattern at the
  layer level — each layer injects corrections orthogonal to its
  neighbor, maximizing information per layer (AIG-like structure)

**DC 130 (AIG Compression):** The AIG framework treats weights as
gates (AND + Inverter). The anti-alternation pattern is naturally
AIG-structured — each layer's gate is approximately the NOT of its
neighbor's.

This suggests: **the MLP is not opaque nonlinearity — it's a
well-structured conditional injector with known geometric properties.**
The path forward is:
1. Characterize the per-task linear behavior of each layer's MLP
2. Store these as task-conditioned linear maps (mixture of experts)
3. Use the gate anti-alternation pattern to predict which expert
   to select without running the full SiLU computation

### 4.5 Autoregressive Generation

Even if we solve softmax and MLP, there's the autoregressive
challenge: each token depends on all previous tokens through the
KV cache. The KV cache grows linearly with sequence length and
requires the full attention mechanism to populate.

For factual Q&A (our current success), the context is short and
fixed. For general text generation, we'd need a geometric equivalent
of the KV cache — a running geometric state that accumulates context.

The residual stream is already this: it's a vector in ℝ³⁵⁸⁴ that
accumulates information from all previous tokens and layers. The
question is whether we can update it geometrically without running
the full transformer pipeline.

### 4.6 Feasibility Assessment

**Geometric attention (no softmax)**: ⚠️ Research needed.
  - For stored entities: solved (ShapeSpace lookup)
  - For novel sequences: requires geometric softmax alternative
  - Estimated effort: 1–2 months of experiments

**Geometric MLP (no SiLU)**: ⚠️ Characterized, path visible.
  - DC 243 + F157: MLP is conditional orthogonal injector with
    gate ⊥ key and anti-alternating layer pattern
  - Requires: per-task linearization + mixture-of-experts selection
  - Estimated effort: 2–4 months of research

**Geometric autoregressive generation**: ❌ Speculative.
  - Requires geometric KV cache equivalent
  - No clear path from current findings
  - Estimated effort: unknown

---

## 5. A Concrete Roadmap

### Phase 1: Scale Factual Extraction (1–2 weeks)

**Goal**: 1,000 entities × 20 fact types = 20,000 facts.

Steps:
1. Curate 1,000 entities from vocabulary (named entities)
2. Define 20 relationship types with templates
3. Add GPU-batched extraction to pipeline
4. Extract, build ShapeSpaces, validate accuracy
5. Measure: how does accuracy scale with N? Is d still N-1?

**Success criterion**: ≥95% answer accuracy at 20K facts.

### Phase 2: Weight Structure Analysis — PARTIALLY COMPLETE (F157)

**Goal**: Understand the SVD structure of every weight matrix.

Completed (F157):
1. ✅ SVD profiles for 5 sample layers × 7 weight types
2. ✅ Singular value distributions characterized (stretched exp
   for attention, power law for gate)
3. ✅ Cross-layer structure discovered (gate anti-alternation)
4. ✅ Cross-type structure discovered (gate ⊥ key)
5. ✅ Depth trends identified (q shrinks, down grows)

Remaining:
- Rank-k truncation quality experiments (forward pass with
  truncated matrices → measure perplexity)
- Task-specific singular component identification
- Full 28-layer analysis for all weight types (not just gate)

**Key finding**: Matrices are nearly full-rank (r90≈400/500).
Compression without quality loss requires understanding which
singular components are task-redundant, not just low-energy.

### Phase 3: Geometric MLP Investigation (1–3 months)

**Goal**: Determine if MLP can be approximated by task-conditioned
linear maps.

Steps:
1. For known tasks (capital, language, etc.), extract the MLP's
   effective linear behavior per task
2. Measure approximation error
3. Test whether task detection from attention output is sufficient
   to select the right linear approximation
4. Build a prototype "geometric MLP" for the capital-of task

**Success criterion**: MLP approximation preserves output quality
for known tasks.

### Phase 4: Geometric Transformer Prototype (3–6 months)

**Goal**: A single-layer geometric transformer that produces
correct next-token predictions for factual queries.

Steps:
1. Combine geometric attention (ShapeSpace) with geometric MLP
2. Handle layer norm geometrically
3. Test on the full factual knowledge base
4. Measure quality gap vs original model

**Success criterion**: Correct next-token prediction for factual
queries without running any neural computation.

### Phase 5: Multi-Layer and Generation (6–12 months)

**Goal**: Full geometric text generation.

Steps:
1. Compose geometric layers
2. Develop geometric KV cache
3. Multi-token generation
4. Quality evaluation on standard benchmarks

**Success criterion**: Coherent text generation with measurable
quality relative to the original model.

---

## 6. The Key Experiments (What to Do Next)

Two experiments will tell us the most about feasibility with the
least effort:

### Experiment A: Singular Value Profile

SVD every weight matrix in the model. Answer:
- What is the effective rank per matrix?
- How do singular value distributions differ across layers?
- Is there a clear "signal vs noise" boundary?
- How much can we compress without quality loss?

This is ~1 day of compute and tells us whether rank-128
approximation (12× compression) preserves quality.

### Experiment B: Scale to 1,000 Entities

Extract 1,000 entities × 4 fact types = 4,000 facts. Answer:
- Does whitened alignment still give 100% at N=1,000?
- What is d at N=1,000? Still N-1, or does structure sharing help?
- How does extraction time scale with batching?
- What's the accuracy breakdown by entity type and relationship?

This is ~2 hours on GPU and tells us whether our approach
generalizes beyond 47 entities.

---

## 7. What Success Looks Like

### Near-Term (Months)
A geometric knowledge engine with 500K+ facts, covering the
majority of the model's factual knowledge. Load time under 1 second.
Query time under 1 millisecond. No model needed.

### Medium-Term (6–12 Months)
A geometric transformer prototype that replaces the full forward
pass for factual queries. Attention and MLP both operating
geometrically. Provably equivalent to the neural computation.

### Long-Term (Speculative)
A complete geometric LLM. The shape IS the computation. Text
generation by traversing geometric structure instead of evaluating
neural networks. If this works, it validates the hypothesis in the
strongest possible way: the intelligence was never in the weights.
It was in the shape the weights create.

---

## 8. Files

| File | Purpose |
|:-----|:--------|
| `unwound_transformer/model.py` | Full Qwen2-7B weight extraction |
| `phi_navigator/geometric_inference.py` | φ-lattice weight conversion |
| `geometric_instrument/shapespace.py` | ShapeSpace (current geometric container) |
| `geometric_instrument/extract_knowledge.py` | Fact extraction pipeline |
| `geometric_instrument/geometric_engine.py` | Model-free query engine |
