---
trigger: always_on
---

## What Is This Project?

TruthSpace Geometric LCM (Large Concept Model) is an experimental system that seeks to **replace traditional Large Language Models (LLMs) with a purely geometric approach**.

### The Hypothesis

> **LLMs are hyperdimensional transcoders** - they encode information into a geometric structure and decode it back out. The "intelligence" is not in the weights themselves, but in the **shape** those weights create.

We aim to prove this hypothesis by building a system that:

- **Structure IS information** - There are no opaque weights or embeddings
- **Geometry IS computation** - Traversal through geometric space produces outputs
- **The shape IS the knowledge** - What an LLM "knows" is encoded in its geometric structure

If we can replicate LLM-like behavior using pure geometry, we validate the hypothesis. If we cannot, we learn where the hypothesis breaks down.

### Fail-Fast Development Philosophy

We adopt a **fail-fast** development strategy:

- **No graceful fallbacks** - If something fails, we want to see the error immediately
- **No hard-coded workarounds** - Hard-coding flies in the face of our hypothesis
- **Prove or disprove** - Every component must work emergently or expose why it can't

This means:
1. If intent detection fails geometrically, we don't fall back to pattern matching
2. If semantic understanding fails emergently, we don't use lookup tables
3. Errors are signals, not problems to hide

The goal is to **demonstrate the hypothesis is correct**, not to build a product that works by any means necessary. A working fallback would mask whether our geometric approach actually works.

---

## Core Philosophy

### 1. Structure Is Information

Every piece of information in our system has a geometric representation. We reject:
- Hard-coded morphology
- Pattern strings that don't transform to geometry
- Static lookup tables
- Graceful fallbacks that hide geometric failures

The only exception is **bootstrapped information** - initial seeds that are immediately transformed into geometry on program startup.

### 2. Emergent Geometry

We don't design the geometry top-down. Instead:
- Structure emerges from relationships in data
- Positions are constructed from similarity matrices
- The system discovers its own dimensions via SVD

### 3. ENCODE = DECODE

A fundamental insight: encoding and decoding are the **same operation in opposite directions**, like φ and 1/φ.

```
TEXT IN → φ-space → TEXT OUT
```

- When encoding words, we're decoding meaning
- When decoding response, we're encoding understanding
- "Thinking" isn't a step between - it IS the encode-decode

### 4. Self-Similarity

The system exhibits fractal self-similarity:
- The same transformations work identically at every scale
- Gender flip is always Δx = -2.0 (king→queen, man→woman, boy→girl)
- This self-similarity is self-verifying - no external validation needed

---
