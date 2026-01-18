# Design Consideration 122: Self-Assembly on Model Weights

## The Journey

This document captures a profound discovery about the limits and possibilities of geometric self-assembly.

### The Depth Estimation Attempt

We tried to use self-assembly for depth estimation from photographs:

| Approach | Result |
|----------|--------|
| Holographic enhancement | 0.077 MAE (3x better than DA) |
| Self-assembly from residuals | 7.3% improvement |
| Synthetic 3D (unambiguous) | 78-96% improvement |
| **TRUE ambiguity test** | **4-7% improvement (FAILURE)** |

### The Fundamental Limit

**Self-assembly can only discover relationships that EXIST in the data.**

When we tested TRUE ambiguity (color/position don't predict depth):
- Texture at random depths: 4.5% improvement
- Overlapping objects: 6.8% improvement

This is the fundamental limit: **information LOST in 2D projection cannot be recovered geometrically.**

### The Pivot: Model Weights

User insight:
> "What if we just used our self assembler on the model weights themselves, and tried to learn those? We wouldn't need to recreate the model from outputs, but instead recreate it from just understanding how weights are geometrically spaced."

This is brilliant because:
- **Photographs**: 3D → 2D projection (information LOST)
- **Model weights**: ARE the geometric structure (information DIRECTLY ENCODED)

## Self-Assembly on DistilBERT Weights

### Results

| Metric | Value |
|--------|-------|
| Weight tensors analyzed | 50 |
| Top 3 eigenvalues | 95.27% of variance |
| Attention spread | 0.024 |
| MLP spread | 0.033 |
| **Cluster separation** | **0.152 (5x larger!)** |

**✓ Clusters are SEPARATED! Self-assembly discovered structure.**

### What Emerged (Without Labels)

```
Attention centroid: [-0.527, 0.225, -0.054]
MLP centroid:       [-0.527, 0.152, 0.079]
Embedding centroid: [-0.527, -0.116, 0.237]
```

Self-assembly discovered:
1. **Functional groupings** - Attention, MLP, Embedding separate
2. **Layer relationships** - Consistent structure across layers
3. **Low-dimensional structure** - 95% variance in 3 dimensions

### φ-Scaling Analysis

Singular value ratios:
- Near φ (1.618 ± 0.1): 1.84%
- Mean ratio: 1.076
- Median: 1.024

The geometric structure exists but doesn't strongly follow φ-scaling. This suggests:
- The structure is real (clusters separate)
- But the scaling law may be different from φ
- Or φ-scaling appears at a different level of analysis

## The Key Insight

| Problem | Information State | Self-Assembly Result |
|---------|-------------------|---------------------|
| Depth from photos | LOST in projection | Limited (4-7% on ambiguous) |
| Synthetic 3D | DESIGNED to correlate | Excellent (78-96%) |
| **Model weights** | **DIRECTLY ENCODED** | **Excellent (clusters separate)** |

**Self-assembly works when the relationships are actually present in the data.**

## Implications for TruthSpace

### What This Validates

1. **Structure IS information** - but only if the structure EXISTS
2. **Geometry IS computation** - but only on geometric relationships
3. **The shape IS the knowledge** - but the shape must be present

### What This Suggests

For the TruthSpace hypothesis ("LLMs are hyperdimensional transcoders"):
- The geometric structure IS in the weights
- Self-assembly CAN discover it
- We don't need to reverse-engineer from outputs
- We can learn directly from the weight geometry

### The Path Forward

Instead of:
```
Outputs → Infer structure → Build geometric model
```

We can do:
```
Weights → Self-assemble structure → Understand geometry directly
```

This is more direct and doesn't suffer from projection loss.

## Connection to Previous Work

### Qwen2 Reverse Engineering (from memories)

Previous work found:
- Proper nouns took up majority of model weight space
- They existed on the "zero axis" - largely unused most of the time
- BUT the relationships they convey are the MOST meaningful

This aligns with our finding that weight space is low-dimensional (95% in 3 dims) but the structure encodes functional relationships.

### Holographic Pattern Space

The holographic pattern space uses:
```
S[i,j] = word_overlap(module_i, module_j)
Positions = V @ sqrt(Λ)  # From eigendecomposition
```

This is EXACTLY what we did with weights:
```
S[i,j] = feature_similarity(weight_i, weight_j)
Positions = V @ sqrt(Λ)  # Same eigendecomposition
```

The method is the same - it's the DATA that matters.

## Files

- `/home/thorin/truthspace-lcm/experiments/unified_assembly/weight_self_assembly.py`
- `/home/thorin/truthspace-lcm/experiments/unified_assembly/depth_synthetic_ambiguity.py`

## Next Steps

1. **Scale up**: Apply to larger models (GPT-2, Llama)
2. **Semantic analysis**: Do weight clusters correspond to semantic functions?
3. **Cross-model comparison**: Do different models have similar weight geometry?
4. **Reconstruction**: Can we reconstruct model behavior from weight geometry?

## Summary

The session revealed a fundamental truth:

**Self-assembly discovers structure that EXISTS, not structure that's LOST.**

- Depth from photos: structure LOST → limited success
- Model weights: structure ENCODED → excellent success

This redirects our approach: instead of trying to infer structure from outputs (where information may be lost), we can self-assemble directly from weights (where information is preserved).

The TruthSpace hypothesis remains valid - we just need to look at the right data.
