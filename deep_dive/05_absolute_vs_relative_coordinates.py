#!/usr/bin/env python3
"""
Deep Dive: Absolute vs Relative Coordinates

The fundamental insight:

SIMILARITY MATRICES give us RELATIVE positions:
- "A is similar to B" tells us A and B are close
- But WHERE are they? We don't know absolutely.
- The eigenspace gives us coordinates, but they're arbitrary
- The DC component is the "average similarity" - a relative baseline

CONSTANT-BASED AXES give us ABSOLUTE positions:
- φ^(-k) levels are mathematically defined
- Each dimension has semantic meaning (action, domain, relation)
- Positions are VERIFIABLE - you can check if they're valid
- No DC component because positions aren't derived from similarity

The old TruthSpace used:
- 12 dimensions with semantic meaning
- φ^level encoding for each primitive
- φ-weighted Euclidean distance

This is like the difference between:
- GPS coordinates (absolute, verifiable)
- "I'm near the coffee shop" (relative, context-dependent)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

# Constants from old TruthSpace
PHI = (1 + np.sqrt(5)) / 2
DIM = 12

# φ-block weights from old implementation
PHI_BLOCK_WEIGHTS = np.array([
    PHI**2, PHI**2, PHI**2, PHI**2,  # Actions: dims 0-3
    1.0, 1.0, 1.0, 1.0,               # Domains: dims 4-7
    PHI**-2, PHI**-2, PHI**-2, PHI**-2  # Relations: dims 8-11
])


def analyze_old_truthspace():
    """Analyze the old TruthSpace coordinate system."""
    print("=" * 70)
    print("OLD TRUTHSPACE: ABSOLUTE COORDINATES")
    print("=" * 70)
    print()
    
    print("Dimension Structure:")
    print("  Dims 0-3:  ACTIONS   (weighted by φ² ≈ 2.618)")
    print("  Dims 4-7:  DOMAINS   (weighted by 1.0)")
    print("  Dims 8-11: RELATIONS (weighted by φ⁻² ≈ 0.382)")
    print()
    
    print("φ-Level Encoding:")
    print("  Level 0: φ⁰ = 1.000")
    print("  Level 1: φ¹ = 1.618")
    print("  Level 2: φ² = 2.618")
    print("  Level 3: φ³ = 4.236")
    print()
    
    # Example encodings
    print("Example Encodings:")
    print()
    
    # "list files" would activate:
    # - LIST (dim 1, level 1) → φ¹ = 1.618
    # - FILE (dim 5, level 0) → φ⁰ = 1.0
    list_files = np.zeros(DIM)
    list_files[1] = PHI ** 1  # LIST
    list_files[5] = PHI ** 0  # FILE
    
    print("  'list files':")
    print(f"    dim 1 (LIST): {list_files[1]:.3f}")
    print(f"    dim 5 (FILE): {list_files[5]:.3f}")
    print(f"    Position: {list_files}")
    print()
    
    # "show disk space" would activate:
    # - READ (dim 1, level 0) → φ⁰ = 1.0
    # - STORAGE (dim 5, level 3) → φ³ = 4.236
    show_disk = np.zeros(DIM)
    show_disk[1] = PHI ** 0   # READ
    show_disk[5] = PHI ** 3   # STORAGE
    
    print("  'show disk space':")
    print(f"    dim 1 (READ): {show_disk[1]:.3f}")
    print(f"    dim 5 (STORAGE): {show_disk[5]:.3f}")
    print(f"    Position: {show_disk}")
    print()
    
    # Distance calculation
    diff = (list_files - show_disk) * PHI_BLOCK_WEIGHTS
    distance = np.linalg.norm(diff)
    
    print(f"  Distance between them: {distance:.3f}")
    print()
    
    print("KEY PROPERTIES:")
    print("  1. Positions are ABSOLUTE - defined by φ^level on fixed dimensions")
    print("  2. Positions are VERIFIABLE - you can check if they're valid")
    print("  3. No DC component - positions aren't derived from similarity")
    print("  4. Semantic dimensions - each axis has meaning")


def analyze_current_approach():
    """Analyze the current similarity-based approach."""
    print()
    print("=" * 70)
    print("CURRENT APPROACH: RELATIVE COORDINATES")
    print("=" * 70)
    print()
    
    from truthspace_lcm.core.chat_pipeline import ChatPipeline, ChatConfig
    
    config = ChatConfig(debug=False)
    pipeline = ChatPipeline(config)
    ks = pipeline.knowledge_space
    
    n = len(ks._mappings)
    positions = np.array([m.position for m in ks._mappings])
    
    print(f"Number of concepts: {n}")
    print(f"Dimensions: {positions.shape[1]}")
    print()
    
    print("Position Statistics:")
    print(f"  Mean position: {positions.mean(axis=0)[:4]}...")
    print(f"  Std per dim: {positions.std(axis=0)[:4]}...")
    print()
    
    # The centroid
    centroid = positions.mean(axis=0)
    
    print("Centroid (the DC component location):")
    print(f"  {centroid[:4]}...")
    print()
    
    # Distance from centroid
    dists_from_centroid = [np.linalg.norm(positions[i] - centroid) for i in range(n)]
    
    print("Distance from centroid:")
    print(f"  Min: {min(dists_from_centroid):.4f}")
    print(f"  Max: {max(dists_from_centroid):.4f}")
    print(f"  Mean: {np.mean(dists_from_centroid):.4f}")
    print()
    
    print("KEY PROPERTIES:")
    print("  1. Positions are RELATIVE - derived from similarity matrix")
    print("  2. Positions are NOT verifiable - no way to check if valid")
    print("  3. DC component exists - captures 'average similarity'")
    print("  4. Dimensions are EMERGENT - no inherent meaning")


def analyze_the_connection():
    """Analyze how the two approaches might connect."""
    print()
    print("=" * 70)
    print("THE CONNECTION: ZETA LINE AS BRIDGE")
    print("=" * 70)
    print()
    
    print("From the Kerr Truth Space discovery:")
    print("  - Neural network weights cluster at φ^(-k) levels")
    print("  - This is the 'zeta line' through truth space")
    print("  - The event horizon at σ = 1/(2φ) separates regimes")
    print()
    
    print("The insight:")
    print("  SIMILARITY MATRICES are pointing at ABSOLUTE positions")
    print("  but we're having trouble finding them because:")
    print("  1. We're using eigendecomposition (relative)")
    print("  2. The DC component obscures the structure")
    print("  3. We don't have the right coordinate system")
    print()
    
    print("The old TruthSpace had the right idea:")
    print("  - φ^level gives ABSOLUTE coordinates")
    print("  - Semantic dimensions give MEANING")
    print("  - The lattice is NAVIGABLE and VERIFIABLE")
    print()
    
    print("The question is:")
    print("  Can we map similarity-based positions to φ-based coordinates?")
    print("  Can we find the ABSOLUTE position that similarity is pointing to?")


def explore_phi_lattice_mapping():
    """Explore mapping similarity positions to φ-lattice."""
    print()
    print("=" * 70)
    print("EXPLORING: φ-LATTICE MAPPING")
    print("=" * 70)
    print()
    
    from truthspace_lcm.core.chat_pipeline import ChatPipeline, ChatConfig
    
    config = ChatConfig(debug=False)
    pipeline = ChatPipeline(config)
    ks = pipeline.knowledge_space
    
    n = len(ks._mappings)
    positions = np.array([m.position for m in ks._mappings])
    
    # The φ-lattice has positions at φ^k for k = ..., -2, -1, 0, 1, 2, ...
    # Can we find which φ-level each dimension is closest to?
    
    print("φ-levels:")
    phi_levels = [PHI ** k for k in range(-5, 6)]
    for k in range(-5, 6):
        print(f"  φ^{k:+d} = {PHI ** k:.6f}")
    print()
    
    print("Mapping eigenspace positions to φ-levels:")
    print()
    
    for dim in range(min(4, positions.shape[1])):
        values = positions[:, dim]
        
        # Find the φ-level closest to each value
        level_counts = {}
        for v in values:
            # Find nearest φ^k
            best_k = None
            best_dist = float('inf')
            for k in range(-10, 10):
                dist = abs(v - PHI ** k)
                if dist < best_dist:
                    best_dist = dist
                    best_k = k
            
            level_counts[best_k] = level_counts.get(best_k, 0) + 1
        
        print(f"Dimension {dim}:")
        print(f"  Value range: [{values.min():.4f}, {values.max():.4f}]")
        print(f"  Most common φ-levels: {sorted(level_counts.items(), key=lambda x: -x[1])[:3]}")
        print()
    
    print("Observation:")
    print("  The eigenspace values don't naturally align to φ-levels.")
    print("  This is because eigenspace is RELATIVE, not ABSOLUTE.")
    print()
    print("  To get ABSOLUTE positions, we need to:")
    print("  1. Define semantic dimensions (like old TruthSpace)")
    print("  2. Encode concepts directly to φ-levels")
    print("  3. Use the φ-lattice for navigation")


def propose_hybrid_approach():
    """Propose a hybrid approach combining both methods."""
    print()
    print("=" * 70)
    print("PROPOSED: HYBRID APPROACH")
    print("=" * 70)
    print()
    
    print("The problem with PURE SIMILARITY:")
    print("  - DC component dominates")
    print("  - Positions are relative, not absolute")
    print("  - No way to verify if a position is 'correct'")
    print()
    
    print("The problem with PURE φ-LATTICE:")
    print("  - Requires predefined primitives")
    print("  - Can't handle novel concepts")
    print("  - Keyword-dependent (not purely geometric)")
    print()
    
    print("HYBRID APPROACH:")
    print()
    print("1. USE φ-LATTICE FOR BOOTSTRAP:")
    print("   - Define semantic dimensions (action, domain, relation)")
    print("   - Place bootstrap concepts at φ^k positions")
    print("   - These are ABSOLUTE anchors")
    print()
    print("2. USE SIMILARITY FOR NOVEL CONCEPTS:")
    print("   - New concepts are placed relative to anchors")
    print("   - But SNAP to nearest φ-lattice point")
    print("   - This gives absolute coordinates for new concepts")
    print()
    print("3. USE ZETA ZEROS AS WAYPOINTS:")
    print("   - Zeta zeros are resonance points between φ-levels")
    print("   - Queries navigate via zeta zeros")
    print("   - This provides the 'natural curve' for navigation")
    print()
    print("The key insight:")
    print("  The φ-lattice provides the COORDINATE SYSTEM")
    print("  Similarity provides the NAVIGATION")
    print("  Zeta zeros provide the WAYPOINTS")


def main():
    print("=" * 70)
    print("DEEP DIVE: ABSOLUTE vs RELATIVE COORDINATES")
    print("=" * 70)
    print()
    print("Your insight: 'Similarity matrices feel like they're pointing at")
    print("some position in space, but we're having a hard time finding")
    print("absolute values for.'")
    print()
    print("This is exactly right. Let's analyze...")
    print()
    
    analyze_old_truthspace()
    analyze_current_approach()
    analyze_the_connection()
    explore_phi_lattice_mapping()
    propose_hybrid_approach()
    
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The DC component problem exists because we're using RELATIVE coordinates
(eigenspace from similarity matrix) instead of ABSOLUTE coordinates
(φ-lattice with semantic dimensions).

The old TruthSpace had the right idea:
- φ^level gives mathematically verifiable positions
- Semantic dimensions give meaning to each axis
- The lattice is navigable without a DC component

The zeta line method shows that values naturally cluster at φ^(-k) levels.
The Kerr Truth Space discovery shows there's a natural curve (the horizon).

The solution may be to:
1. Return to φ-based absolute coordinates
2. Use similarity to NAVIGATE, not to DEFINE positions
3. Use zeta zeros as waypoints on the natural curve

This would eliminate the DC component problem entirely because
positions would be ABSOLUTE, not relative to a similarity centroid.
""")


if __name__ == "__main__":
    main()
