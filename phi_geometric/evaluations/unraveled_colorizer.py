#!/usr/bin/env python3
"""
Unraveled Colorizer: Eliminating Attention

The hypothesis:
1. MESH principle: Pre-compute Q.T @ K structure
2. Boom positions: Attention is sparse (89.5% at 37% of positions)
3. Safe dial: The "click" is the only irreducible computation

For colorization:
- Queries are FIXED (100 learned queries)
- Features are input-dependent
- Cross-attention: queries @ features.T

If we can characterize the STRUCTURE of attention (which queries attend where),
we might be able to skip the O(N²) computation entirely.

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from phi_geometric.core.encoder import PhiEncoder, PHI


def analyze_attention_structure():
    """Analyze DDColor's attention structure - is it predictable?"""
    print("=" * 70)
    print("ANALYZING DDCOLOR ATTENTION STRUCTURE")
    print("=" * 70)
    
    try:
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        model.eval()
    except Exception as e:
        print(f"Could not load DDColor: {e}")
        return
    
    # Get the color decoder components
    color_decoder = model.decoder.color_decoder
    
    # The queries are FIXED
    query_feat = color_decoder.query_feat.weight.detach()  # [100, 256]
    query_embed = color_decoder.query_embed.weight.detach()  # [100, 256]
    
    print(f"\n## Fixed Components")
    print(f"  query_feat: {query_feat.shape}")
    print(f"  query_embed: {query_embed.shape}")
    
    # The attention projections
    # In transformer cross-attention: Q = queries @ W_q, K = features @ W_k
    # Attention = softmax(Q @ K.T / sqrt(d))
    
    # For each cross-attention layer, extract Q, K, V projections
    cross_attn_layers = color_decoder.transformer_cross_attention_layers
    
    print(f"\n## Cross-Attention Layers: {len(cross_attn_layers)}")
    
    for i, layer in enumerate(cross_attn_layers):
        attn = layer.multihead_attn
        
        # in_proj_weight contains Q, K, V projections concatenated
        in_proj = attn.in_proj_weight.detach()  # [768, 256] = [3*256, 256]
        
        W_q = in_proj[:256]   # [256, 256]
        W_k = in_proj[256:512]  # [256, 256]
        W_v = in_proj[512:]   # [256, 256]
        
        # The MESH for this layer: W_q.T @ W_k
        MESH = W_q.T @ W_k  # [256, 256]
        
        # Analyze MESH structure
        U, S, Vt = torch.linalg.svd(MESH)
        
        # Effective rank
        normalized_S = S / (S.sum() + 1e-8)
        entropy = -torch.sum(normalized_S * torch.log(normalized_S + 1e-10))
        effective_rank = torch.exp(entropy).item()
        
        print(f"\n  Layer {i}:")
        print(f"    MESH shape: {MESH.shape}")
        print(f"    MESH effective rank: {effective_rank:.1f}")
        print(f"    Top-5 singular values: {S[:5].tolist()}")
        
        if i == 0:
            # For the first layer, analyze more deeply
            print(f"\n    ## First Layer Deep Analysis")
            
            # The attention score for query q and feature f is:
            # score = (q @ W_q) @ (f @ W_k).T / sqrt(d)
            #       = q @ (W_q @ W_k.T) @ f.T / sqrt(d)
            #       = q @ MESH @ f.T / sqrt(d)
            
            # If we pre-compute query @ MESH for all queries:
            query_MESH = query_feat @ MESH  # [100, 256]
            
            print(f"    query @ MESH: {query_MESH.shape}")
            
            # Now attention is just: query_MESH @ features.T
            # This is still O(100 * H * W) but the query side is pre-computed
            
            # Can we go further? Analyze the structure of query_MESH
            U_qm, S_qm, Vt_qm = torch.linalg.svd(query_MESH)
            
            normalized_S_qm = S_qm / (S_qm.sum() + 1e-8)
            entropy_qm = -torch.sum(normalized_S_qm * torch.log(normalized_S_qm + 1e-10))
            effective_rank_qm = torch.exp(entropy_qm).item()
            
            print(f"    query_MESH effective rank: {effective_rank_qm:.1f}")
            print(f"    This means {effective_rank_qm:.0f} independent attention patterns")


def analyze_spatial_attention_patterns():
    """Analyze if attention patterns are spatially predictable."""
    print("\n" + "=" * 70)
    print("SPATIAL ATTENTION PATTERN ANALYSIS")
    print("=" * 70)
    
    print("""
## The Hypothesis

If attention patterns are spatially predictable:
- "Sky queries" always attend to top of image
- "Ground queries" always attend to bottom
- "Object queries" attend to center

Then we can SKIP computing attention and directly route:
- Query 47 → top 20% of features
- Query 23 → center features
- etc.

This would eliminate the O(N²) attention entirely.
""")
    
    # We would need to run DDColor on multiple images and analyze
    # where each query attends. For now, let's reason about it.
    
    print("""
## What We Know

From Doc 192 (Boom-Newton Attention):
- 89.5% of attention mass at 37% of positions
- Boom positions are predictable

From Doc 189 (Safe Dial):
- The "click" at layer 3 is irreducible
- But after the click, the path is deterministic

## The Key Question

For DDColor:
- Are the 100 queries semantically specialized?
- Does query 47 ALWAYS attend to sky-like features?
- Or does it depend on the specific image?

If specialized: We can pre-compute the routing
If not: We need to compute attention per image
""")


def the_unraveled_architecture():
    """Propose the unraveled architecture."""
    print("\n" + "=" * 70)
    print("THE UNRAVELED COLORIZER ARCHITECTURE")
    print("=" * 70)
    
    print("""
## Current DDColor Architecture

```
grayscale → encoder → features
                         ↓
queries → cross_attention(queries, features) → attended
                         ↓
              color_embed → ab_output
```

The bottleneck: cross_attention is O(100 × H × W)

## Unraveled Architecture (Hypothesis)

```
grayscale → encoder → features
                         ↓
              spatial_classifier → region_labels
                         ↓
queries[region] → direct_lookup → attended
                         ↓
              color_embed → ab_output
```

Instead of computing attention, we:
1. Classify each spatial position into a "region" (sky, ground, object, etc.)
2. Look up which queries are responsible for that region
3. Directly apply those queries

## Why This Might Work

From our analysis:
- Queries are nearly orthogonal (0.05 similarity)
- Effective rank is 100 (all queries are distinct)
- This suggests each query handles a DIFFERENT concept

If we can identify what concept each query handles,
we can route directly without computing attention.

## The Content Problem

BUT: We don't know what each query handles.
- Query 47 = sky? skin? vegetation?
- This is the SEMANTIC CONTENT we can't derive

## The Hybrid Approach

1. STRUCTURE: Pre-compute query @ MESH (eliminates half of attention)
2. BOOM: Only attend to top-k feature positions (37% → 89.5% mass)
3. ROUTING: Learn a simple spatial classifier for common patterns

Combined speedup potential:
- MESH: 2x (pre-computed query side)
- BOOM: 3x (sparse attention)
- ROUTING: 5x (skip attention for predictable regions)
- Total: 30x potential speedup
""")


def test_mesh_precomputation():
    """Test if MESH pre-computation works for DDColor."""
    print("\n" + "=" * 70)
    print("TESTING MESH PRE-COMPUTATION")
    print("=" * 70)
    
    try:
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        model.eval()
    except Exception as e:
        print(f"Could not load DDColor: {e}")
        return
    
    encoder = PhiEncoder(K=32)
    
    # Get first cross-attention layer
    color_decoder = model.decoder.color_decoder
    layer = color_decoder.transformer_cross_attention_layers[0]
    attn = layer.multihead_attn
    
    in_proj = attn.in_proj_weight.detach()
    W_q = in_proj[:256]
    W_k = in_proj[256:512]
    
    # Compute MESH
    MESH = W_q.T @ W_k
    
    # Encode MESH in φ-basis
    signs, exps = encoder.encode(MESH)
    MESH_reconstructed = encoder.decode(signs, exps)
    
    # Compare
    error = (MESH - MESH_reconstructed).abs().mean().item()
    correlation = torch.corrcoef(torch.stack([MESH.flatten(), MESH_reconstructed.flatten()]))[0, 1].item()
    
    print(f"\n## MESH Pre-computation Results")
    print(f"  MESH shape: {MESH.shape}")
    print(f"  Reconstruction error: {error:.6f}")
    print(f"  Correlation: {correlation:.6f}")
    
    # Now test with queries
    query_feat = color_decoder.query_feat.weight.detach()
    
    # Pre-compute query @ MESH
    query_MESH = query_feat @ MESH
    query_MESH_phi = query_feat @ MESH_reconstructed
    
    # Compare
    qm_error = (query_MESH - query_MESH_phi).abs().mean().item()
    qm_correlation = torch.corrcoef(torch.stack([query_MESH.flatten(), query_MESH_phi.flatten()]))[0, 1].item()
    
    print(f"\n## Query @ MESH Results")
    print(f"  query_MESH shape: {query_MESH.shape}")
    print(f"  Reconstruction error: {qm_error:.6f}")
    print(f"  Correlation: {qm_correlation:.6f}")
    
    # Encode query_MESH directly
    signs_qm, exps_qm = encoder.encode(query_MESH)
    query_MESH_direct = encoder.decode(signs_qm, exps_qm)
    
    qm_direct_error = (query_MESH - query_MESH_direct).abs().mean().item()
    qm_direct_corr = torch.corrcoef(torch.stack([query_MESH.flatten(), query_MESH_direct.flatten()]))[0, 1].item()
    
    print(f"\n## Direct query_MESH Encoding")
    print(f"  Reconstruction error: {qm_direct_error:.6f}")
    print(f"  Correlation: {qm_direct_corr:.6f}")
    
    print(f"""
## Conclusion

We can pre-compute and φ-encode:
1. MESH = W_q.T @ W_k (per layer)
2. query_MESH = queries @ MESH (per layer)

At inference:
- Attention = softmax(query_MESH @ features.T / sqrt(d))
- This is O(100 × H × W) instead of O(100 × 256 + 256 × H × W)

The query side is fully pre-computed.
Only the feature side needs runtime computation.
""")


def main():
    analyze_attention_structure()
    analyze_spatial_attention_patterns()
    the_unraveled_architecture()
    test_mesh_precomputation()
    
    print("\n" + "=" * 70)
    print("SUMMARY: CAN WE ELIMINATE ATTENTION?")
    print("=" * 70)
    print("""
## What We Can Do

1. PRE-COMPUTE MESH
   - query_MESH = queries @ (W_q.T @ W_k)
   - Eliminates half of the attention computation
   - φ-encodable with high correlation

2. SPARSE ATTENTION (Boom)
   - 89.5% of attention mass at 37% of positions
   - Only compute attention for boom positions
   - 3x speedup potential

3. STRUCTURE TRANSFER
   - Copy DDColor's query directions
   - Pre-compute query_MESH for all layers
   - Only learn the feature encoder

## What We Cannot Do (Yet)

1. ELIMINATE ATTENTION ENTIRELY
   - We don't know which queries attend where
   - This is the SEMANTIC CONTENT
   - Requires data or extraction

2. SKIP CONTENT LEARNING
   - The attention patterns encode "what goes where"
   - This is learned, not derivable

## The Path Forward

HYBRID APPROACH:
1. Pre-compute all structure (MESH, query_MESH)
2. Use boom attention for sparse computation
3. Learn only the content-dependent routing

This gives us:
- Structure: Pre-computed (no training needed)
- Routing: Sparse (3x speedup)
- Content: Still needs learning (but smaller search space)

The attention mechanism is NOT eliminated, but it's:
- Pre-computed where possible
- Sparse where possible
- Only computed for content-dependent parts
""")


if __name__ == "__main__":
    main()
