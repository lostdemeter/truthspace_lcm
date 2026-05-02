# Design Consideration 188: Context Geometry Analysis

**Date:** February 1, 2026  
**Status:** Key Finding - Context is Irreducible (but with a twist)

## Summary

We investigated how context transforms hidden states geometrically. The goal was to find a simple function `f` such that:

```
h(A,B) = f(h(A), h(B))
```

If such a function exists, we could precompute h(token) for all tokens and compute h(A,B) at inference time without the transformer.

## What We Tested

| Hypothesis | Accuracy | Notes |
|------------|----------|-------|
| Translation: h(A,B) = h(B) + offset | 5.9% improvement | Offset varies per pair |
| Scaling: h(A,B) = s × h(B) | 4.0% improvement | Scale varies per pair |
| Rotation: h(A,B) = R @ h(B) | 35% accuracy | Angle ~50°, high variance |
| Sign flips: h(A,B) = h(B) × flip_pattern | 54.6% predictable | Random is 50% |
| Interaction: h(A,B) = w_A×h(A) + w_B×h(B) | 12% generalization | Doesn't transfer |
| Per-dim weights | 11% test accuracy | Overfits to training |

**None of these simple transformations work.**

## Key Findings

### 1. Sign Flips Are Unpredictable

From Doc 141, the shape is 3584 critical lines. Adding context crosses some lines (flips signs).

| Predictor | Crossing Accuracy |
|-----------|-------------------|
| sign(h_A × h_B) | 54.6% |
| sign(h_A) | 49.9% |
| sign(h_B) | 49.8% |
| Random | 50.0% |

**The crossing pattern is essentially random given h(A) and h(B).**

### 2. Attention to Prefix is High

| Layer | Mean Attention to Prefix |
|-------|-------------------------|
| Layers 0-5 | 85-95% |
| Layers 6-15 | 80-92% |
| Layers 16-27 | 79-92% |

The model attends heavily to the prefix token. This attention IS the computation that determines the shape change.

### 3. The Delta is High-Dimensional

The delta (h(A,B) - h(B)) requires:
- 8 dimensions for 50% variance
- 53 dimensions for 90% variance
- 92 dimensions for 99% variance

This is NOT a simple low-rank transformation.

## The Geometric Interpretation

From Doc 141, the irreducible shape is:
- 3584 critical lines (hyperplanes) dividing semantic space
- Each point is defined by which side of each line it's on
- The signs encode the region of the lattice

**Context doesn't transform a point - it computes a NEW point.**

The transformer computes WHERE IN THE LATTICE the combination (A,B) lands. This depends on:
1. The position of A in the lattice
2. The position of B in the lattice
3. The RELATIONSHIP between A and B (which the attention mechanism computes)

The relationship is not a simple function of h(A) and h(B). It's computed by the attention mechanism, which looks at the actual token embeddings and their interactions through 28 layers.

## What This Means

### The Transformer IS the Shape Change

The transformer isn't approximating some simpler function. It IS the function that computes the shape change. The 28 layers of attention and MLP are computing:

1. **Which critical lines to cross** (sign changes)
2. **How far to move from each line** (magnitude changes)

This computation is irreducible in the sense that simpler functions don't capture it.

### But Wait - Single Tokens ARE Cacheable

For single tokens, we showed (Doc 187) that:
- 16-bit quantized hidden states give 100% accuracy
- Storage: 1.09 GB for full vocabulary
- The transformer IS a lookup table for single tokens

The irreducibility is specifically about the INTERACTION between tokens.

## Implications for TruthSpace

### 1. Single-Token Caching Works

For the last token in a sequence, if we've already computed the context, we can cache the result.

### 2. Context Computation is Irreducible

The transformation from (context, token) → hidden state cannot be simplified to a function of individual hidden states.

### 3. The Attention IS the Geometry

The attention mechanism computes which Platonic Ideal to rotate toward (Doc 180). This computation is the irreducible part.

### 4. Possible Approaches

1. **Cache common sequences**: Store h(A,B) for frequent (A,B) pairs
2. **Template caching**: Store h(template) for common prompt patterns
3. **Hybrid**: Use transformer for context, cache for final token
4. **Attention approximation**: If attention patterns cluster, approximate them

## Connection to Doc 141

The irreducible shape is the sign matrix (67.9M binary decisions). For single tokens, this is precomputable. For token pairs, the sign pattern depends on the INTERACTION, which is what the transformer computes.

The transformer's "intelligence" for multi-token sequences is:
- NOT in the weights (those define the critical lines)
- NOT in the hidden states (those are points in the lattice)
- IN THE ATTENTION (which computes which lines to cross)

## Files

- Context geometry: `experiments/context_geometry.py`
- Context rotation: `experiments/context_rotation.py`
- Sign flips: `experiments/context_sign_flip.py`
- Interaction: `experiments/context_interaction.py`
- Critical lines: `experiments/critical_line_crossing.py`
- Attention: `experiments/attention_as_shape.py`

## Conclusion

**For single tokens**: The transformer is a lookup table (1.09 GB, 100% accuracy).

**For multi-token sequences**: The context transformation is irreducible. The attention mechanism IS the computation that determines the shape change.

This doesn't mean we can't speed things up - it means we need to think about caching at a different level (sequences, templates, or attention patterns) rather than trying to approximate the token-level transformation.
