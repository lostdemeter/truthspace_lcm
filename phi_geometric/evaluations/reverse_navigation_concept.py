#!/usr/bin/env python3
"""
Reverse Navigation: From Destination to Origin

The key insight:
- We HAVE the destination (trained weights on φ-lattice)
- Training is just the PATH to that destination
- If we know where we're going, can we skip the journey?

Traditional training: Random init → Gradient descent → Converged weights
Reverse navigation:   Converged weights → Analyze structure → Derive init principle

The question: What makes the destination special?
- Why do weights converge to THESE lattice points?
- What's the attractor basin?
- Can we identify the destination without walking there?

Author: TruthSpace LCM Project
Date: February 6, 2026
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat/experiments/ddcolor_reference')

from phi_geometric.core.encoder import PhiEncoder, PHI


def analyze_convergence_destination():
    """Analyze what makes the converged weights special."""
    print("=" * 70)
    print("ANALYZING THE DESTINATION")
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
    
    # Collect all weight statistics
    all_weights = []
    all_exponents = []
    layer_stats = []
    
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        
        w = param.detach().cpu()
        all_weights.append(w.flatten())
        
        # Get φ-exponents
        signs, exps = encoder.encode(w)
        levels = (exps.float() - encoder.bias) / encoder.K
        all_exponents.append(levels.flatten())
        
        # Per-layer stats
        layer_stats.append({
            'name': name,
            'shape': tuple(w.shape),
            'mean_level': levels.mean().item(),
            'std_level': levels.std().item(),
            'sparsity': (w.abs() < 1e-6).float().mean().item(),
        })
    
    all_weights = torch.cat(all_weights)
    all_exponents = torch.cat(all_exponents)
    
    print(f"\n## Global Statistics")
    print(f"  Total parameters: {len(all_weights):,}")
    print(f"  Mean φ-level: {all_exponents.mean():.2f}")
    print(f"  Std φ-level: {all_exponents.std():.2f}")
    print(f"  Mode φ-level: {all_exponents.mode().values.item():.2f}")
    
    # Distribution of exponents
    print(f"\n## φ-Level Distribution")
    for level in range(-15, 5):
        count = ((all_exponents >= level) & (all_exponents < level + 1)).sum().item()
        pct = count / len(all_exponents) * 100
        bar = "█" * int(pct)
        print(f"  φ^{level:3d}: {pct:5.1f}% {bar}")
    
    return layer_stats, all_exponents


def compare_random_vs_trained():
    """Compare random initialization to trained weights."""
    print("\n" + "=" * 70)
    print("RANDOM vs TRAINED: What's Different?")
    print("=" * 70)
    
    encoder = PhiEncoder(K=32)
    
    # Random initialization (typical Xavier/He)
    random_weights = torch.randn(1000, 256) * 0.01
    r_signs, r_exps = encoder.encode(random_weights)
    r_levels = (r_exps.float() - encoder.bias) / encoder.K
    
    print(f"\n## Random Init (Xavier-like)")
    print(f"  Mean φ-level: {r_levels.mean():.2f}")
    print(f"  Std φ-level: {r_levels.std():.2f}")
    
    # Trained weights (from DDColor)
    try:
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        
        # Get color decoder weights
        query_feat = model.decoder.color_decoder.query_feat.weight.detach()
        t_signs, t_exps = encoder.encode(query_feat)
        t_levels = (t_exps.float() - encoder.bias) / encoder.K
        
        print(f"\n## Trained Weights (DDColor queries)")
        print(f"  Mean φ-level: {t_levels.mean():.2f}")
        print(f"  Std φ-level: {t_levels.std():.2f}")
        
        # The key difference
        print(f"\n## The Difference")
        print(f"  Random is centered at φ^{r_levels.mean():.1f}")
        print(f"  Trained is centered at φ^{t_levels.mean():.1f}")
        print(f"  Shift: {t_levels.mean() - r_levels.mean():.1f} levels")
        
    except Exception as e:
        print(f"Could not load DDColor: {e}")


def analyze_structure_not_values():
    """The structure might be more important than specific values."""
    print("\n" + "=" * 70)
    print("STRUCTURE vs VALUES")
    print("=" * 70)
    
    print("""
Key insight: Training doesn't find ARBITRARY values.
It finds values that satisfy STRUCTURAL CONSTRAINTS.

What constraints?
1. Orthogonality: Queries should be distinct
2. Coverage: Queries should span the color space
3. Sparsity: Attention should be selective
4. Smoothness: Similar inputs → similar outputs

These constraints DEFINE the destination.
The destination is not arbitrary - it's the FIXED POINT
of the constraint satisfaction process.

If we can identify the constraints, we can:
1. Solve for the fixed point directly
2. Skip the gradient descent journey
3. Initialize AT the destination
""")
    
    # Analyze orthogonality of trained queries
    try:
        from ddcolor import DDColor
        from huggingface_hub import PyTorchModelHubMixin
        
        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)
        
        model = DDColorHF.from_pretrained('piddnad/ddcolor_paper_tiny')
        
        query_feat = model.decoder.color_decoder.query_feat.weight.detach()
        
        # Normalize
        q_norm = query_feat / query_feat.norm(dim=1, keepdim=True)
        
        # Compute similarity matrix
        sim = q_norm @ q_norm.T
        
        # Off-diagonal similarities (should be low for orthogonal)
        off_diag = sim - torch.eye(100)
        
        print(f"\n## Query Orthogonality")
        print(f"  Mean off-diagonal similarity: {off_diag.abs().mean():.4f}")
        print(f"  Max off-diagonal similarity: {off_diag.abs().max():.4f}")
        print(f"  Queries are {'nearly orthogonal' if off_diag.abs().mean() < 0.1 else 'NOT orthogonal'}")
        
        # Analyze coverage (do queries span the space?)
        U, S, Vt = torch.linalg.svd(query_feat)
        
        print(f"\n## Query Coverage (SVD)")
        print(f"  Effective rank: {(S > S.max() * 0.01).sum().item()}")
        print(f"  Top singular value: {S[0]:.2f}")
        print(f"  10th singular value: {S[9]:.2f}")
        print(f"  Ratio (spread): {S[9] / S[0]:.2f}")
        
        # The structure IS the constraint satisfaction
        print(f"""
## The Structural Constraints

1. ORTHOGONALITY: Queries have low mutual similarity ({off_diag.abs().mean():.4f})
   → Each query captures a DISTINCT color concept
   
2. COVERAGE: Effective rank is {(S > S.max() * 0.01).sum().item()}
   → Queries SPAN the color space
   
3. SCALE: Weights are at φ^-9 level
   → Appropriate magnitude for attention softmax

These constraints DEFINE the destination.
Training is just one way to find it.
""")
        
    except Exception as e:
        print(f"Could not load DDColor: {e}")


def propose_intentional_training():
    """Propose how to make training intentional."""
    print("\n" + "=" * 70)
    print("INTENTIONAL TRAINING: Monitoring the Journey")
    print("=" * 70)
    
    print("""
## The Proposal

Instead of blind gradient descent, we:
1. MONITOR the trajectory on the φ-lattice
2. IDENTIFY when weights approach known destinations
3. ACCELERATE by jumping to the destination

## Implementation

```python
class IntentionalTrainer:
    def __init__(self, model, known_destinations):
        self.model = model
        self.destinations = known_destinations  # From analyzed models
        self.encoder = PhiEncoder(K=32)
    
    def training_step(self, batch):
        # Normal forward/backward
        loss = self.compute_loss(batch)
        loss.backward()
        
        # MONITOR: Where are weights going?
        for name, param in self.model.named_parameters():
            signs, exps = self.encoder.encode(param)
            current_position = (signs, exps)
            
            # Check distance to known destinations
            for dest_name, dest_pos in self.destinations.items():
                distance = self.lattice_distance(current_position, dest_pos)
                
                if distance < threshold:
                    # ACCELERATE: Jump to destination
                    param.data = self.encoder.decode(*dest_pos)
                    print(f"{name} converged to {dest_name}!")
        
        self.optimizer.step()
```

## The Key Insight

Training is searching for a fixed point.
If we know the fixed points (from analyzed models),
we can recognize when we're approaching one and JUMP there.

This is like:
- GPS navigation: You know the destination, take the shortest path
- vs. Random walk: Eventually you might get there

## What We Need

1. A LIBRARY of known destinations
   - Extract from trained models (DDColor, Qwen, etc.)
   - Catalog the structural patterns
   
2. A DISTANCE METRIC on the φ-lattice
   - How close is current position to destination?
   - When is it "close enough" to jump?
   
3. A RECOGNITION system
   - Which destination is this weight approaching?
   - Match by structure, not exact values
""")


def the_reverse_navigation_principle():
    """The core principle of reverse navigation."""
    print("\n" + "=" * 70)
    print("THE REVERSE NAVIGATION PRINCIPLE")
    print("=" * 70)
    
    print("""
## Traditional View

    Random Init → Training → Converged Weights
         ?      →    ?     →       ?
    
    We don't know where we're going until we get there.

## Reverse Navigation View

    Converged Weights → Analyze → Structural Constraints → Direct Init
          KNOWN       →  KNOWN  →        DERIVE         →   COMPUTE
    
    We KNOW the destination. Work backward to find the principle.

## The Process

1. COLLECT destinations
   - DDColor: 100 queries at φ^-9, nearly orthogonal, span color space
   - Qwen: MESH matrices with specific eigenstructure
   - DA2: Funnel weights with specific compression ratio
   
2. ANALYZE commonalities
   - All weights at φ^-9 (same scale)
   - Orthogonality constraints
   - Coverage constraints
   - Sparsity patterns
   
3. DERIVE the principle
   - What makes these destinations STABLE?
   - What's the attractor basin?
   - Can we compute the fixed point directly?
   
4. INITIALIZE at destination
   - Skip training entirely
   - Or: Train with destination awareness

## The Hypothesis

The destination is not arbitrary.
It's the UNIQUE fixed point of the constraint system.

If we can formalize the constraints, we can SOLVE for the destination.
This is not machine learning - it's constraint satisfaction.

## Example: Color Queries

Constraints:
- 100 queries in 256-dim space
- Nearly orthogonal (distinct concepts)
- Span the color space (coverage)
- At φ^-9 scale (attention compatibility)

Solution:
- Take 100 orthogonal vectors in 256-dim
- Scale to φ^-9
- Done. No training needed.

But wait - which 100 orthogonal vectors?
THIS is where semantic knowledge enters.
The STRUCTURE is derivable. The MEANING requires data.

## The Refined Understanding

    Structure (derivable)     ×    Meaning (requires data)
    ─────────────────────         ────────────────────────
    - Orthogonality               - Which colors
    - Coverage                    - Which semantics
    - Scale (φ^-9)                - Which associations
    - Sparsity                    - Which contexts

We can derive the FORM of the solution.
The CONTENT must come from somewhere.

But even this is progress:
- We know the solution is 100 orthogonal vectors at φ^-9
- We just need to find the RIGHT 100 vectors
- This is a much smaller search space than all possible weights
""")


def main():
    layer_stats, all_exponents = analyze_convergence_destination()
    compare_random_vs_trained()
    analyze_structure_not_values()
    propose_intentional_training()
    the_reverse_navigation_principle()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The destination (trained weights) satisfies structural constraints:
1. Orthogonality: Queries are distinct
2. Coverage: Queries span the space
3. Scale: Weights at φ^-9
4. Sparsity: Selective attention

These constraints DEFINE the destination.
Training is just one way to find it.

Reverse navigation:
1. Analyze trained models to find constraints
2. Solve constraints directly (skip training)
3. Or: Monitor training, jump to destination when close

The gap is not "how to train" but "what to train TO".
We have the destination. Now we need to understand WHY it's the destination.
""")


if __name__ == "__main__":
    main()
