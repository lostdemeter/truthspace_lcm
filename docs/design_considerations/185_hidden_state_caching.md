# Design Consideration 185: Hidden State Caching and Compression

**Date:** February 1, 2026  
**Status:** Validated

## Summary

We investigated what makes hidden states entity-specific and how to cache them efficiently. The key finding: **hidden states can be compressed 512x using basis decomposition**, but they **cannot be predicted or interpolated** for new entities.

## The Investigation

### What We Tried

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Simple rotation (Doc 180) | 16.7% | Transformer does more than rotation |
| Linear transform from embedding | 0% generalization | Overfits to training data |
| Hidden state interpolation | 16.7% | Entities not geometrically close |
| Sign-only storage | 13% | Signs alone insufficient |
| Basis decomposition | **100%** | Works for known entities |

### The Breakthrough: Basis Decomposition

Hidden states can be decomposed as:
```
H = mean + Σ c_i × ψ_i
```

Where:
- `mean`: Shared mean hidden state (per relationship type)
- `ψ_i`: Basis functions (learned from SVD)
- `c_i`: Entity-specific coefficients

**Results:**

| Components | Coefficient Bits | Accuracy | Storage/Entity |
|------------|------------------|----------|----------------|
| k=15 | 4-bit | **100%** | **7 bytes** |
| k=15 | 8-bit | 100% | 15 bytes |
| k=15 | 16-bit | 100% | 30 bytes |
| Full (3584 dims) | 8-bit | 100% | 3,584 bytes |

**Compression: 512x** (from 3,584 bytes to 7 bytes)

### The Limitation: No Generalization

The basis learned from training data does NOT generalize to new entities:

| Training Size | Training Accuracy | Generalization |
|---------------|-------------------|----------------|
| 12 entities | 100% | 67% |
| 30 entities | 100% | 30% |

**Why?** Each entity has a unique hidden state that lies in a different direction. The basis captures variance in TRAINING entities, not ALL entities.

## Connection to Tetromino Hypothesis (Doc 162)

The Tetromino Hypothesis showed:
- Weights exist on a constrained φ-lattice
- Only ~300 unique (level, sign_pattern) combinations
- 81.6% of deltas within ±2 levels

We tested if hidden states have similar structure:
- **φ-lattice reconstruction (k=16): 100% accuracy** ✓
- **Delta encoding: Only 18% within ±10** ✗

Hidden states ARE on the φ-lattice, but their deltas are NOT as constrained as weights. This makes sense: weights are trained to be structured, but hidden states are computed outputs.

## Connection to Bulge Discovery (Doc 180)

The Bulge Discovery showed:
- Trajectories = Geodesic + Bulge
- Bulge shape is universal
- 10 coefficients capture 87.5% variance
- 2,867x compression

Hidden states have similar structure:
- H = mean + Σ c_i × ψ_i
- 15 basis functions capture 100% decoding accuracy
- 512x compression

**The difference:** Bulge shape is universal across trajectories, but hidden state basis is NOT universal across entities.

## The Architecture

Since we MUST store per-entity, the practical approach is:

```
Query → Extract entity → In cache?
  │
  ├─ YES → Reconstruct from coefficients (instant, no transformer)
  │        H = mean + Σ c_i × ψ_i
  │        Decode token from H
  │
  └─ NO  → Compute via transformer
           Learn new basis function if needed
           Store coefficients in cache
```

### Storage Requirements

| Scale | Naive (8-bit) | Basis (7 bytes) | Savings |
|-------|---------------|-----------------|---------|
| 1M entities | 3.6 GB | 7 MB | 512x |
| 10M entities | 36 GB | 70 MB | 512x |
| 100M entities | 360 GB | 700 MB | 512x |

Plus shared storage:
- Mean hidden state: 14 KB (float32)
- Basis functions (k=15): 215 KB (float32)
- Total shared: ~230 KB per relationship type

### Self-Improving Cache

As more entities are queried:
1. Transformer computes hidden state
2. Hidden state is projected onto current basis
3. If projection error is high, basis is expanded
4. Coefficients are stored

Over time, the basis becomes more universal, and more entities can be reconstructed without recomputing.

## Implications

### For TruthSpace LCM

This validates a key insight:
> **The transformer's "knowledge" is in the hidden states, not the weights.**

We can't skip the 28 layers, but we CAN cache their output. The hidden state IS the answer - we just need to decode it.

### For Practical Systems

A hybrid approach is optimal:
1. **Hot cache**: Most common entities with basis coefficients
2. **Warm cache**: Less common entities with full hidden states
3. **Cold compute**: Rare entities via transformer

### For Understanding

The hidden state decomposition reveals:
- ~36% of hidden state is entity-specific (residual/mean ratio)
- This entity-specific part is spread across ALL dimensions
- It cannot be predicted from the entity embedding alone
- The transformer's 28 layers compute something irreducible

## Files

- Analysis: `experiments/hidden_state_compression.py`
- φ-lattice: `experiments/hidden_state_phi_lattice.py`
- Factorization: `experiments/hidden_state_factorization.py`
- Signs: `experiments/hidden_state_signs.py`
- Basis: `experiments/hidden_state_basis.py`
- Large scale: `experiments/hidden_state_basis_large.py`

## Conclusion

Hidden states can be cached with **512x compression** using basis decomposition, but they **cannot be predicted** for new entities. The practical path forward is a self-improving cache that grows with usage, eventually covering most common queries.

**Structure IS information. The hidden state IS the answer. We just need to store it efficiently.**
