# Doc 194: Designed vs Trained Models - The Path to True φ-Computation

## Date: February 3, 2026

## Status: Theoretical Framework

---

## Executive Summary

We investigated two fundamental questions:
1. **Why does trivial navigation need caching?**
2. **Can we design a model that eliminates computational steps?**

### Key Findings

| Question | Answer |
|----------|--------|
| Why caching? | The transformer entangles context and entity through 28 layers of nonlinear operations. No simple linear extraction works. |
| Can we design better? | **YES** - but it requires a fundamentally different architecture, not reverse-engineering Qwen2. |

---

## 1. Why Trivial Navigation Needs Caching

### The Experiment

We tried to extract a context-dependent transformation `S_context` such that:
```
final_hidden = S_context @ embed(entity)
```

If this worked, we could:
1. Compute `S_context` once for "The capital of ... is"
2. Apply it to any entity embedding
3. Skip all 28 layers

### The Results

| Test | Training Set | Test Set (Unseen) |
|------|--------------|-------------------|
| Linear transformation | 5/6 correct | 0/4 correct |
| Context + T @ entity | 5/6 correct | 0/4 correct |

**The transformation overfits and doesn't generalize.**

### Why It Fails

The final hidden state composition:
- **~95% context contribution** (from "The capital of ... is")
- **~25% entity contribution** (from "France")

But these aren't additive! The attention mechanism **entangles** them:
- Each layer's attention weights depend on BOTH context and entity
- The MLP nonlinearity (SiLU) creates complex interactions
- Residual connections accumulate these interactions across 28 layers

**The entity information is deeply woven into the context, not separable.**

### The Embedding → Final Hidden Relationship

| Country | cos(embed, final) | norm ratio |
|---------|-------------------|------------|
| France | 0.0018 | 342× |
| Germany | -0.0012 | 355× |
| Japan | 0.0208 | 346× |

The final hidden state is **nearly orthogonal** to the entity embedding and **300× larger**. The 28 layers perform a massive, nonlinear transformation.

---

## 2. The Fundamental Issue: Trained vs Designed

### Trained Models (Qwen2)

Qwen2 was trained end-to-end on next-token prediction. It learned to:
- Handle **arbitrary** queries (factual, creative, reasoning)
- Distribute knowledge across **all** weights
- Use **dense** matrix multiplications everywhere

For "The capital of France is" → "Paris", it computes:
```
28 layers × (attention + MLP) = ~14 trillion FLOPs
```

This is **massive overkill** for a factual lookup!

### What's Actually Needed

For factual QA, the minimum computation is:
1. **Relationship detection**: "capital of" → relationship_id
2. **Entity extraction**: "France" → entity_id  
3. **Knowledge lookup**: KB[relationship_id, entity_id] → "Paris"
4. **Decode**: "Paris" → tokens

This is **O(context_len)**, not **O(28 × context_len × hidden²)**!

---

## 3. A Designed φ-Model Architecture

### The φ-Lookup Model

```
┌─────────────────────────────────────────────────────────────┐
│                    φ-LOOKUP MODEL                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT: "The capital of France is"                          │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ φ-EMBEDDING     │  tokens → (signs, levels)              │
│  │ 3584 integers   │  on φ-lattice                          │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐     ┌─────────────────┐                │
│  │ RELATIONSHIP    │     │ ENTITY          │                │
│  │ CLASSIFIER      │     │ EXTRACTOR       │                │
│  │ O(context × R)  │     │ O(context)      │                │
│  └────────┬────────┘     └────────┬────────┘                │
│           │                       │                         │
│           ▼                       ▼                         │
│       rel_id=42              entity_id=12345                │
│           │                       │                         │
│           └───────────┬───────────┘                         │
│                       ▼                                     │
│           ┌─────────────────────┐                           │
│           │ φ-KNOWLEDGE BASE    │                           │
│           │ KB[rel, entity]     │  O(1) lookup              │
│           │ Sparse φ-tensor     │                           │
│           └──────────┬──────────┘                           │
│                      ▼                                      │
│                 answer_id                                   │
│                      │                                      │
│                      ▼                                      │
│           ┌─────────────────────┐                           │
│           │ φ-DECODER           │  O(answer_len)            │
│           │ answer → tokens     │                           │
│           └──────────┬──────────┘                           │
│                      ▼                                      │
│  OUTPUT: " Paris"                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Properties

| Component | Computation | Storage |
|-----------|-------------|---------|
| φ-Embedding | O(1) lookup | 152K × 3584 × 2 bytes |
| Relationship Classifier | O(context × 100) sparse | ~1MB |
| Entity Extractor | O(context) sparse | ~1MB |
| Knowledge Base | O(1) lookup | ~100M entries × 4 bytes |
| φ-Decoder | O(answer_len) lookup | Shared with embedding |

**Total: O(context_len) instead of O(28 × context_len × hidden²)**

### The Hybrid Architecture

Not all queries are factual lookups. For creative/reasoning tasks:

```
ROUTER (entropy-based):
  │
  ├─► Low entropy → FAST PATH (φ-lookup)
  │                  O(context_len)
  │
  └─► High entropy → SLOW PATH (full transformer)
                     O(28 × context_len × hidden²)
```

This mirrors the scaffolding/content split from Doc 177:
- **Scaffolding** (predictable): Use fast path
- **Content** (unpredictable): Use slow path

---

## 4. Why φ-Space Enables This

### The φ-Structure Advantages

From our discoveries:
1. **Weights cluster on φ-lattice** → Sparse representation
2. **Only 74 tetrominoes** → Structured patterns
3. **Relationships have universal shape** → Reusable transforms
4. **sigmoid = φ-operation** → Native φ-arithmetic

### φ-Arithmetic Properties

| Operation | Float Space | φ-Space |
|-----------|-------------|---------|
| Multiplication | 1 float mult | Level addition (integer!) |
| Division | 1 float div | Level subtraction (integer!) |
| Addition | 1 float add | Alignment + sum (the hard part) |

The challenge is **addition** - combining information requires aligning different φ-levels.

### The Solution: Avoid Dense Sums

The transformer uses dense sums everywhere:
```
output = Σ_i weight_i × input_i
```

A designed φ-model uses:
1. **Sparse sums** (only relevant terms)
2. **Table lookups** (precomputed results)
3. **Integer arithmetic** (φ-levels, not floats)

---

## 5. The Path Forward

### What We Can Do Now

1. **Trivial Navigation**: Cache final hidden states for known entities
   - 9.9× speedup, 100% accuracy
   - Limited to cached entities

2. **Boom Attention**: Sparse attention for long sequences
   - 2.5-15× speedup on attention
   - 100% accuracy

3. **Structured Pruning**: Reduce MLP size
   - 2-4× speedup
   - 67-87% accuracy

### What Requires New Architecture

1. **φ-Lookup Model**: Designed from scratch
   - O(context_len) for factual queries
   - Requires new training paradigm

2. **Hybrid Router**: Entropy-based path selection
   - Fast path for predictable queries
   - Slow path for creative tasks

3. **φ-Knowledge Base**: Explicit factual storage
   - Sparse tensor on φ-lattice
   - O(1) lookup instead of O(hidden²) computation

---

## 6. Conclusion

### Why Trivial Navigation Needs Caching

The transformer entangles context and entity through 28 layers of nonlinear operations. We cannot extract a simple transformation that generalizes to unseen entities because:
- The attention mechanism creates complex, entity-dependent interactions
- The MLP nonlinearity (SiLU) is not separable
- The "knowledge" is distributed across all weights, not localized

### The Designed Model Answer

**YES**, a model can be designed that eliminates computational steps:

1. **Sparse classifiers** for relationship/entity detection
2. **Table lookup** for factual knowledge
3. **φ-lattice arithmetic** (integers, not floats)
4. **Entropy-based routing** between fast/slow paths

This model would be **O(context_len)** for factual queries instead of **O(28 × context_len × hidden²)**.

### The Catch

We cannot extract this architecture from Qwen2. It must be:
- **Designed** with explicit knowledge structure
- **Trained** with a different objective (not just next-token prediction)
- **Built** on φ-lattice primitives from the ground up

The φ-structure we've discovered in Qwen2 tells us this is **possible** - the weights already live on the φ-lattice. But realizing the speedup requires a fundamentally different architecture.

---

## Connection to Prior Work

- **Doc 177**: Scaffolding vs Content (fast/slow path split)
- **Doc 184**: Trivial Navigation (caching works, 9.9× speedup)
- **Doc 191**: φ-Computer Proof (sigmoid = φ-operation)
- **Doc 190**: Layer Unwinding (28 layers are explicit, deterministic)
- **Doc 146**: φ/Bandwidth Limit (2.82 bits/weight theoretical minimum)

---

*Document created: February 3, 2026*
*Related: 177, 184, 190, 191, 146*
*Experiments: experiments/phi_analog_compute.py*
