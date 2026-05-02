#!/usr/bin/env python3
"""
Extract DDColor's Learned Atoms

DDColor has 100 "color queries" - learned positions in embedding space
that capture real-world color distributions.

These ARE the atoms we need. We don't need to hand-code them.
We can extract them from the pretrained model.

The question: Can we characterize these atoms geometrically?
If so, we can either:
1. Use them directly (steal the knowledge)
2. Derive them from geometric principles
3. Simulate data that would produce them

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from phi_geometric.core.encoder import PhiEncoder, PHI, LN_PHI

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_color_queries():
    """Extract DDColor's 100 color queries."""
    print("=" * 70)
    print("EXTRACTING DDCOLOR'S LEARNED ATOMS")
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
        print("DDColor loaded")
    except Exception as e:
        print(f"Could not load DDColor: {e}")
        return None
    
    # Extract color decoder components
    color_decoder = model.decoder.color_decoder
    
    # The 100 color queries
    query_feat = color_decoder.query_feat.weight.detach().cpu()  # [100, 256]
    query_embed = color_decoder.query_embed.weight.detach().cpu()  # [100, 256]
    
    print(f"\n## Color Queries")
    print(f"  query_feat: {query_feat.shape}")
    print(f"  query_embed: {query_embed.shape}")
    
    # Analyze the queries
    print(f"\n## Query Statistics")
    print(f"  query_feat mean: {query_feat.mean():.4f}")
    print(f"  query_feat std: {query_feat.std():.4f}")
    print(f"  query_feat range: [{query_feat.min():.4f}, {query_feat.max():.4f}]")
    
    # φ-level analysis
    encoder = PhiEncoder(K=32)
    signs, exps = encoder.encode(query_feat)
    levels = (exps.float() - encoder.bias) / encoder.K
    
    print(f"\n## φ-Lattice Structure")
    print(f"  Mean φ-level: {levels.mean():.2f}")
    print(f"  Std φ-level: {levels.std():.2f}")
    print(f"  Peak φ-level: {levels.mode().values.mean():.2f}")
    
    # The color embedding MLP - this maps queries to colors
    color_embed = color_decoder.color_embed
    print(f"\n## Color Embedding MLP")
    for name, param in color_embed.named_parameters():
        print(f"  {name}: {param.shape}")
    
    # Pass queries through color embedding to get effective colors
    with torch.no_grad():
        queries = query_feat.unsqueeze(0)  # [1, 100, 256]
        color_output = color_embed(queries)  # [1, 100, 256]
    
    print(f"\n## Color Output")
    print(f"  Shape: {color_output.shape}")
    print(f"  Mean: {color_output.mean():.4f}")
    print(f"  Std: {color_output.std():.4f}")
    
    # Analyze the structure of the 100 queries
    # Do they cluster? Are they on a manifold?
    print(f"\n## Query Clustering (SVD)")
    U, S, Vt = torch.linalg.svd(query_feat)
    
    cumsum = torch.cumsum(S**2, dim=0) / (S**2).sum()
    for k in [1, 2, 5, 10, 20, 50]:
        print(f"  Top-{k} singular values explain: {cumsum[k-1]*100:.1f}%")
    
    # Effective rank
    normalized_S = S / S.sum()
    entropy = -torch.sum(normalized_S * torch.log(normalized_S + 1e-10))
    effective_rank = torch.exp(entropy)
    print(f"  Effective rank: {effective_rank:.1f}")
    
    return {
        "query_feat": query_feat,
        "query_embed": query_embed,
        "color_output": color_output,
        "S": S,
        "effective_rank": effective_rank.item(),
    }


def analyze_what_queries_encode(results):
    """Analyze what the 100 queries actually encode."""
    if results is None:
        return
    
    print("\n" + "=" * 70)
    print("WHAT DO THE QUERIES ENCODE?")
    print("=" * 70)
    
    query_feat = results["query_feat"]
    S = results["S"]
    
    # The queries are 100 points in 256-dimensional space
    # But the effective rank is much lower
    print(f"\n## Dimensionality")
    print(f"  Nominal: 100 queries × 256 dims = 25,600 values")
    print(f"  Effective rank: {results['effective_rank']:.1f}")
    print(f"  Actual information: ~{results['effective_rank']:.0f} dimensions")
    
    # This means the 100 queries lie on a ~20-dimensional manifold
    # We can represent them with ~20 basis vectors
    
    # Compute the top-k approximation
    U, S, Vt = torch.linalg.svd(query_feat)
    
    for k in [5, 10, 20]:
        approx = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]
        error = torch.norm(query_feat - approx) / torch.norm(query_feat)
        print(f"  Rank-{k} approximation error: {error*100:.2f}%")
    
    # The key insight: we can represent 100 queries with ~20 basis vectors
    # This is the "simulated data" - the basis vectors ARE the training signal
    
    print(f"\n## The Minimum Representation")
    k = 20  # Use rank-20 approximation
    
    # The basis vectors (Vt[:k]) are the "atoms"
    # The coefficients (U[:, :k] @ diag(S[:k])) are the "positions"
    
    basis = Vt[:k, :]  # [20, 256] - the 20 basis vectors
    coeffs = U[:, :k] @ torch.diag(S[:k])  # [100, 20] - coefficients for each query
    
    print(f"  Basis vectors: {basis.shape} = {basis.numel()} values")
    print(f"  Coefficients: {coeffs.shape} = {coeffs.numel()} values")
    print(f"  Total: {basis.numel() + coeffs.numel()} values")
    print(f"  Original: {query_feat.numel()} values")
    print(f"  Compression: {query_feat.numel() / (basis.numel() + coeffs.numel()):.1f}x")
    
    return {
        "basis": basis,
        "coeffs": coeffs,
        "k": k,
    }


def can_we_derive_the_basis():
    """Can we derive the basis vectors from geometric principles?"""
    print("\n" + "=" * 70)
    print("CAN WE DERIVE THE BASIS?")
    print("=" * 70)
    
    print("""
The 20 basis vectors encode the "directions" in color space.

Hypothesis: These directions correspond to:
1. Luminance (bright ↔ dark)
2. Warm ↔ cool (red-yellow vs blue-green)
3. Saturation (vivid ↔ muted)
4. Semantic categories (sky, vegetation, skin, etc.)
5. Texture types (smooth, textured, edge)
6. Context (indoor, outdoor, natural, artificial)

If we can identify these directions, we can:
- Derive them from first principles
- Or simulate data that spans these directions

The "training data" we need is:
- Examples that span each direction
- The directions themselves are the knowledge

This is like PCA: the principal components ARE the knowledge.
The training data just helps us find them.

Can we find them without training?
- Use color theory (opponent colors, etc.)
- Use semantic categories (our 19 atoms, expanded)
- Use geometric structure (φ-lattice positions)
""")
    
    # The key insight: the basis vectors are STRUCTURED
    # They're not random - they encode meaningful directions
    # We might be able to derive them from first principles
    
    print("\n## The Path Forward")
    print("""
Option 1: STEAL the basis
    - Extract the 20 basis vectors from DDColor
    - Use them directly in our minimal colorizer
    - This gives us DDColor's knowledge without training
    
Option 2: DERIVE the basis
    - Identify the semantic directions (luminance, warm/cool, etc.)
    - Construct basis vectors from first principles
    - This is "training" without data
    
Option 3: SIMULATE the data
    - Generate synthetic images that span the color space
    - Use these to "train" (find the basis)
    - This is training with simulated data
    
The key insight: we don't need REAL data.
We need data that SPANS the relevant directions.
Synthetic data that covers the color space is sufficient.
""")


def main():
    results = extract_color_queries()
    if results:
        basis_results = analyze_what_queries_encode(results)
        can_we_derive_the_basis()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
DDColor's 100 color queries lie on a ~20-dimensional manifold.

The minimum representation is:
    20 basis vectors × 256 dims = 5,120 values
    100 coefficients × 20 dims = 2,000 values
    Total: 7,120 values (vs 25,600 original)
    
This is still more than our 21-parameter model, but it's
the ACTUAL knowledge DDColor learned.

The question is: can we derive these 20 basis vectors
from geometric principles, or do we need to extract them?

If we can derive them, we have "training without data".
If we extract them, we have "knowledge transfer".

Either way, the path is clear:
    1. Get the 20 basis vectors (derive or extract)
    2. Use them to colorize (not our naive 19 atoms)
    3. The result should match DDColor
""")


if __name__ == "__main__":
    main()
