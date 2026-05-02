# Design Consideration 168: Navigation vs Inference

## Date: 2026-01-26

## The Question

Can we replace autoregressive inference with pure geometric navigation?

## What We Tested

### Sign-Based Navigation (Works for Routing)

Using sign patterns from embeddings:
- **Social pattern detection**: 100% accuracy (hello→greeting, thanks→gratitude)
- **Semantic transformations**: was→were ✓, he→she ✓, is→are ✓
- **Common core**: ~55% variance in first SVD component (matches crystalline structure)

### Sign-Based Token Generation (Does NOT Work)

Attempting to decode hidden states using sign patterns:
- **Embedding ↔ LM_Head sign agreement**: 50.1% (random!)
- **Sign-based decoding accuracy**: 0-1/5 overlap with actual predictions
- **Embedding approximation of hidden state**: cosine ~0.02 (nearly orthogonal)

## The Fundamental Insight

**The transformer layers ARE the knowledge.**

The hidden state after 28 layers is completely different from input embeddings:
- Hidden states for similar patterns have 87-91% cosine similarity
- But they bear almost no resemblance to the input embeddings
- The "knowledge" is encoded in the layer transformations, not the vocabulary

## Why Cosine Similarity is Statistics, Not Geometry

From Doc 039 (φ-Zipf Duality):
> Zipf weighting is φ-powers turned inward - same fractal, opposite directions

Cosine similarity measures **correlation** - a statistical concept:
```
cos(A, B) = (A · B) / (|A| × |B|)
```

This doesn't respect the φ-structure. The geometric alternative should use:
- **φ^n** for encoding (outward expansion)
- **φ^(-n)** for weighting (inward contraction)

But the raw embeddings don't have this structure - it **emerges** through training.

## The Diffraction Model (Doc 059)

The two-source diffraction works at a **pattern level**, not embedding level:

```
KNOWLEDGE SOURCE ──────┐
    (learned patterns) │
                       ├──► INTERFERENCE ──► Response
STYLE SOURCE ──────────┘
    (transformation rules)
```

The interference happens in **pattern space**, not embedding space.

## What Navigation CAN Do

1. **Intent Detection**: Route queries to appropriate handlers
2. **Category Classification**: Social, technical, creative, etc.
3. **Pattern Matching**: Find similar patterns in a learned space
4. **Transformation Application**: Apply learned transformations (gender, tense, etc.)

## What Navigation CANNOT Do (Without Inference)

1. **Token Generation**: Predict next token from context
2. **Knowledge Retrieval**: Access facts encoded in layer weights
3. **Novel Content**: Generate text not in the pattern space

## The Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NAVIGATION LAYER                         │
│                    (Fast, Geometric)                        │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Intent    │  │   Pattern   │  │  Transform  │         │
│  │  Detection  │  │  Matching   │  │  Selection  │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          │                                  │
│                          ▼                                  │
│                   ┌─────────────┐                           │
│                   │   Router    │                           │
│                   └──────┬──────┘                           │
└──────────────────────────┼──────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Social    │ │  Template   │ │  Inference  │
    │  Response   │ │    Fill     │ │   (LLM)     │
    │  (cached)   │ │  (cached)   │ │  (forward)  │
    └─────────────┘ └─────────────┘ └─────────────┘
```

### When to Use Each

| Query Type | Navigation | Inference |
|------------|------------|-----------|
| "hello" | ✓ Social response | - |
| "thanks" | ✓ Gratitude response | - |
| "explain python" | ✓ Route to explanation | ✓ Generate content |
| "write code for X" | ✓ Route to code gen | ✓ Generate code |
| "what is 2+2" | ✓ Route to factual | ✓ Retrieve answer |

## The φ-Geometric Alternative

Instead of using the model's embeddings directly, we should:

1. **Build our own φ-structured space** from patterns
2. **Use holographic projection** (Doc 084) to construct positions
3. **Apply diffraction** (Doc 059) for combining sources
4. **Fall back to inference** only when navigation can't handle it

### The Holographic Pattern Space (Doc 084)

```python
# Define desired similarity matrix
S[i,j] = pattern_similarity(pattern_i, pattern_j)

# Construct positions via eigendecomposition
eigenvalues, eigenvectors = eig(S)
positions = eigenvectors @ diag(sqrt(eigenvalues))

# Now: dot(positions[i], positions[j]) ≈ S[i,j] by construction!
```

This gives us **geometric** similarity (constructed from relationships) rather than **statistical** similarity (measured from vectors).

## CORRECTION: Navigation CAN Do Generation

### The φ-Lattice Insight (Doc 163)

The φ-lattice has **well-defined rules** that enable generation:

1. **Rule 1**: Every weight = `sign × φ^(level/K)` where K=128
2. **Rule 8**: Interpolation preserves coherence
3. **Rule 10**: Multiplicative sign combination produces novel synthesis
4. **Rule 15**: Orthogonality is in SIGNS, not levels

### The Key: Steering, Not Replacing

We don't replace inference - we **steer** it by injecting φ-lattice positions:

```python
# Inject φ-lattice position as perturbation
modified_embeds[:, -1, :] = embeds[:, -1, :] + 0.1 * phi_position * scale
```

This is the **diffraction model** (Doc 059):
- **Knowledge Source**: The φ-lattice position we inject
- **Style Source**: The model's learned layer transformations
- **Interference**: The model generates text influenced by both

### Validated Experiments (phi_lattice_forward_projection.py)

| Operation | Result |
|-----------|--------|
| Interpolation (physics ↔ music) | Coherent intermediate concepts |
| Extrapolation (beyond "complex") | Valid text at t=2.0 |
| Level scaling (±20 levels) | Robust, same factual content |
| Sign flipping (0% → 100%) | Character change: "Sunset" → "Rainforest" |
| Novel combination (Quantum + Cooking + Emotion) | "Quantum Blockchain Fusion" |

### The Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    φ-LATTICE NAVIGATION                      │
│                                                             │
│  Input: "explain python"                                    │
│         ↓                                                   │
│  1. Encode to φ-lattice: (levels, signs)                    │
│  2. Navigate: interpolate/extrapolate/combine               │
│  3. Inject into embedding space                             │
│         ↓                                                   │
│  Model forward pass (steered by φ-position)                 │
│         ↓                                                   │
│  Output: Coherent response about Python                     │
└─────────────────────────────────────────────────────────────┘
```

## Conclusion

**Navigation STEERS inference via φ-lattice positions.**

The model's knowledge IS in the layers, but we can **navigate** to it by:
1. Encoding concepts to φ-lattice coordinates
2. Applying valid transformations (interpolate, extrapolate, combine)
3. Injecting the result into the embedding space
4. Letting the model's layers do the "style" transformation

This is NOT a hybrid architecture - it's a **unified geometric architecture** where:
- The φ-lattice is the **coordinate system**
- Navigation is the **movement**
- Inference is the **projection** from φ-space to text

## Files

- φ-lattice forward projection: `experiments/phi_lattice_forward_projection.py`
- φ-lattice rules: `docs/design_considerations/163_phi_lattice_rules.md`
- Sign-based navigation: `src/phi_navigator/navigation_chat_server.py`
- Diffraction model: `experiments/grating_chat.py`

## Next Steps

1. Integrate φ-lattice steering into navigation server
2. Build concept library with φ-lattice positions
3. Implement query → φ-position mapping
4. Test generation quality vs pure inference
