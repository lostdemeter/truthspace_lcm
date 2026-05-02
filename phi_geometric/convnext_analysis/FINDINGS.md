# ConvNeXt Reverse Engineering Findings

**Date**: February 6, 2026

## Summary

ConvNeXt is a hierarchical vision encoder that transforms images into semantic features.
It is the "intelligence" behind DDColor's colorization ability.

---

## Architecture

### Structure
```
ConvNeXt-Tiny (27.8M parameters)
├── downsample_layers (5.6%) - Initial stem + stage transitions
└── stages (94.4%) - Main processing
    ├── Stage 0: 3 blocks, 96 channels, 128x128
    ├── Stage 1: 3 blocks, 192 channels, 64x64
    ├── Stage 2: 9 blocks, 384 channels, 32x32  ← Most computation
    └── Stage 3: 3 blocks, 768 channels, 16x16
```

### ConvNeXt Block
```
input
  ↓
Depthwise Conv 7x7 (spatial mixing)
  ↓
LayerNorm
  ↓
Linear 1x1 (expand 4x)
  ↓
GELU
  ↓
Linear 1x1 (reduce 4x)
  ↓
+ input (residual)
```

---

## Weight Analysis

### SVD Zipf Exponents

| Layer Type | Mean α | Target (1/φ) | Match? |
|------------|--------|--------------|--------|
| Depthwise Conv | 1.08 | 0.618 | ❌ No |
| Pointwise Conv | 0.42 | 0.618 | ❌ No |

**Finding**: ConvNeXt weights do NOT follow φ-Zipf pattern.

- Depthwise convs have **steeper** decay (α ≈ 1.08)
- Pointwise convs have **slower** decay (α ≈ 0.42)

This is different from:
- Attention heads (α ≈ 0.65, close to φ-Zipf)
- MLP layers (α ≈ 0.12, very slow decay)

---

## Feature Analysis

### Semantic Clustering Quality

| Stage | Channels | Resolution | Cluster Ratio |
|-------|----------|------------|---------------|
| 0 | 96 | 128x128 | 1.23 |
| 1 | 192 | 64x64 | 1.80 |
| 2 | 384 | 32x32 | **2.22** |
| 3 | 768 | 16x16 | 1.48 |

**Finding**: Stage 2 has the best semantic clustering.

### Geometric vs ConvNeXt Correlation

| Stage | Mean Correlation | Max Correlation |
|-------|------------------|-----------------|
| 0 | -0.007 | 0.30 |
| 1 | 0.007 | 0.38 |
| 2 | -0.002 | 0.36 |
| 3 | -0.004 | 0.42 |

**Finding**: Geometric features have **ZERO correlation** with ConvNeXt features.

---

## What ConvNeXt Learns

### Stage 0 (Low-level)
- Edges, textures, local patterns
- Similar to Gabor filters but learned
- Our geometric encoder CAN approximate this

### Stage 1 (Mid-level)
- Texture combinations, simple shapes
- Beginning of object parts
- Partially approximable geometrically

### Stage 2 (High-level) ← KEY STAGE
- Object parts, semantic regions
- "This looks like grass", "This looks like sky"
- **NOT approximable without learning**

### Stage 3 (Abstract)
- Object categories, scene understanding
- Global context and relationships
- Requires learned representations

---

## The Semantic Gap

| ConvNeXt | Geometric Encoder |
|----------|-------------------|
| "This is sky" → blue | "Smooth, bright, top" → ??? |
| "This is grass" → green | "Textured, mid-tone, bottom" → ??? |
| "This is skin" → flesh | "Smooth, mid-tone, face-shaped" → ??? |

ConvNeXt has learned to recognize **objects**.
Geometric features only capture **appearance**.

---

## Can We Replace ConvNeXt Geometrically?

### What's Needed

1. **Object recognition** - Know what things ARE, not just how they look
2. **Semantic clustering** - Group pixels by meaning, not just texture
3. **Hierarchical refinement** - Build up from edges to objects

### Possible Approaches

1. **Use pretrained vision models** (CLIP, DINO, SAM)
   - Already have semantic understanding
   - Can be probed for geometric structure

2. **Build semantic vocabulary from scratch**
   - Define object categories geometrically
   - Associate with color distributions
   - Requires massive manual effort

3. **Self-supervised learning**
   - Train on auxiliary tasks (contrastive, masked prediction)
   - Learn semantic structure from data
   - Still requires training

4. **Hybrid approach**
   - Use geometric features for low-level (Stage 0-1)
   - Use pretrained features for high-level (Stage 2-3)

---

## Conclusion

**ConvNeXt cannot be replaced geometrically without some form of learning.**

The key insight: ConvNeXt's "intelligence" comes from ImageNet pretraining,
where it learned to recognize 1000 object categories. This semantic knowledge
cannot be derived from first principles - it must be learned from data.

### Options Going Forward

1. **Accept that encoders need training** - Focus on making training geometric
2. **Use pretrained encoders** - Probe them for geometric structure
3. **Build simpler tasks first** - Edge detection, texture classification
4. **Investigate self-supervised methods** - Can we learn semantics geometrically?

---

## Files

- `01_architecture_analysis.py` - ConvNeXt structure
- `02_weight_analysis.py` - SVD and Zipf analysis
- `03_feature_probing.py` - Semantic clustering analysis
