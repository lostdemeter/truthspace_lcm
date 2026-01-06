#!/usr/bin/env python3
"""
Deep Dive: Implementing φ-Lattice Coordinates

The key insight from the old TruthSpace:
- Positions are ABSOLUTE, defined by φ^level on semantic dimensions
- No DC component because positions aren't derived from similarity
- Mathematically verifiable - you can check if a position is valid

This script explores how to implement this for knowledge matching.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

PHI = (1 + np.sqrt(5)) / 2


class PhiLattice:
    """
    A coordinate system based on φ-levels.
    
    Each dimension has semantic meaning.
    Positions are at φ^k for integer k.
    """
    
    def __init__(self, dimensions: dict):
        """
        Args:
            dimensions: Dict mapping dimension index to semantic name
                        e.g., {0: 'action', 1: 'domain', 2: 'specificity'}
        """
        self.dimensions = dimensions
        self.ndim = len(dimensions)
        
        # Precompute φ-levels
        self.phi_levels = {k: PHI ** k for k in range(-10, 11)}
    
    def snap_to_lattice(self, position: np.ndarray) -> np.ndarray:
        """Snap a position to the nearest φ-lattice point."""
        snapped = np.zeros_like(position)
        for i in range(len(position)):
            # Find nearest φ^k
            best_k = 0
            best_dist = float('inf')
            for k, level in self.phi_levels.items():
                dist = abs(position[i] - level)
                if dist < best_dist:
                    best_dist = dist
                    best_k = k
            snapped[i] = self.phi_levels[best_k]
        return snapped
    
    def position_to_levels(self, position: np.ndarray) -> list:
        """Convert position to φ-level indices."""
        levels = []
        for i in range(len(position)):
            best_k = 0
            best_dist = float('inf')
            for k, level in self.phi_levels.items():
                dist = abs(position[i] - level)
                if dist < best_dist:
                    best_dist = dist
                    best_k = k
            levels.append(best_k)
        return levels
    
    def levels_to_position(self, levels: list) -> np.ndarray:
        """Convert φ-level indices to position."""
        return np.array([PHI ** k for k in levels])
    
    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance in φ-space."""
        return np.linalg.norm(a - b)
    
    def is_valid_position(self, position: np.ndarray, tolerance: float = 0.01) -> bool:
        """Check if position is on the φ-lattice."""
        snapped = self.snap_to_lattice(position)
        return np.allclose(position, snapped, atol=tolerance)


def test_phi_lattice():
    """Test the φ-lattice coordinate system."""
    print("=" * 70)
    print("φ-LATTICE COORDINATE SYSTEM")
    print("=" * 70)
    print()
    
    # Define semantic dimensions
    dimensions = {
        0: 'domain',      # What area? (physics, identity, greeting)
        1: 'specificity', # How specific? (general → specific)
        2: 'formality',   # How formal? (casual → formal)
    }
    
    lattice = PhiLattice(dimensions)
    
    print("Dimensions:")
    for i, name in dimensions.items():
        print(f"  {i}: {name}")
    print()
    
    # Define some concepts at φ-lattice positions
    concepts = {
        'physics': lattice.levels_to_position([2, 1, 0]),      # domain=φ², spec=φ¹, form=φ⁰
        'science': lattice.levels_to_position([1, 0, 0]),      # domain=φ¹, spec=φ⁰, form=φ⁰
        'identity': lattice.levels_to_position([0, 0, 1]),     # domain=φ⁰, spec=φ⁰, form=φ¹
        'greeting': lattice.levels_to_position([-1, -1, -1]),  # domain=φ⁻¹, spec=φ⁻¹, form=φ⁻¹
        'hello': lattice.levels_to_position([-1, -2, -2]),     # More specific greeting
    }
    
    print("Concept Positions (φ-levels):")
    for name, pos in concepts.items():
        levels = lattice.position_to_levels(pos)
        print(f"  {name}: levels={levels}, position={pos}")
    print()
    
    # Check validity
    print("Position Validity:")
    for name, pos in concepts.items():
        valid = lattice.is_valid_position(pos)
        print(f"  {name}: {'✓ valid' if valid else '✗ invalid'}")
    print()
    
    # Test with a non-lattice position
    noisy_pos = concepts['physics'] + np.random.randn(3) * 0.1
    print(f"Noisy position: {noisy_pos}")
    print(f"  Valid: {lattice.is_valid_position(noisy_pos)}")
    print(f"  Snapped: {lattice.snap_to_lattice(noisy_pos)}")
    print()
    
    # Distance matrix
    print("Distance Matrix:")
    names = list(concepts.keys())
    for i, name1 in enumerate(names):
        row = []
        for name2 in names:
            d = lattice.distance(concepts[name1], concepts[name2])
            row.append(f"{d:.2f}")
        print(f"  {name1:10s}: {' '.join(row)}")
    
    return lattice, concepts


def compare_with_eigenspace():
    """Compare φ-lattice with eigenspace approach."""
    print()
    print("=" * 70)
    print("COMPARISON: φ-LATTICE vs EIGENSPACE")
    print("=" * 70)
    print()
    
    from truthspace_lcm.core.chat_pipeline import ChatPipeline, ChatConfig
    
    config = ChatConfig(debug=False)
    pipeline = ChatPipeline(config)
    ks = pipeline.knowledge_space
    
    # Get eigenspace positions
    eigen_positions = np.array([m.position for m in ks._mappings])
    
    print("EIGENSPACE:")
    print(f"  Dimensions: {eigen_positions.shape[1]}")
    print(f"  Position range: [{eigen_positions.min():.4f}, {eigen_positions.max():.4f}]")
    print(f"  Centroid: {eigen_positions.mean(axis=0)[:4]}...")
    print()
    
    # Create a φ-lattice with same number of dimensions
    ndim = eigen_positions.shape[1]
    dimensions = {i: f'dim_{i}' for i in range(ndim)}
    lattice = PhiLattice(dimensions)
    
    # Try to snap eigenspace positions to φ-lattice
    print("Snapping eigenspace positions to φ-lattice:")
    
    for i in range(min(5, len(ks._mappings))):
        eigen_pos = eigen_positions[i]
        snapped = lattice.snap_to_lattice(eigen_pos)
        levels = lattice.position_to_levels(eigen_pos)
        
        print(f"  Concept {i}:")
        print(f"    Eigen: {eigen_pos[:4]}...")
        print(f"    Levels: {levels[:4]}...")
        print(f"    Snapped: {snapped[:4]}...")
        print()
    
    print("Observation:")
    print("  Eigenspace positions are in the range [0.1, 0.5]")
    print("  This corresponds to φ^(-2) to φ^(-1) range")
    print("  The positions are COMPRESSED into a small region")
    print()
    print("  In contrast, φ-lattice positions span φ^(-10) to φ^(+10)")
    print("  This gives much more DYNAMIC RANGE for discrimination")


def explore_semantic_dimensions():
    """Explore what semantic dimensions might look like."""
    print()
    print("=" * 70)
    print("SEMANTIC DIMENSIONS FOR KNOWLEDGE MATCHING")
    print("=" * 70)
    print()
    
    print("Proposed semantic dimensions (inspired by old TruthSpace):")
    print()
    print("  Dim 0: DOMAIN")
    print("    - What area of knowledge?")
    print("    - φ^3: Physics, Math, Science")
    print("    - φ^2: Technology, Programming")
    print("    - φ^1: General knowledge")
    print("    - φ^0: Meta/Identity")
    print("    - φ^-1: Social/Greeting")
    print()
    print("  Dim 1: SPECIFICITY")
    print("    - How specific is the concept?")
    print("    - φ^3: Very specific (quantum mechanics)")
    print("    - φ^2: Specific (physics)")
    print("    - φ^1: General (science)")
    print("    - φ^0: Very general (knowledge)")
    print()
    print("  Dim 2: INTENT")
    print("    - What kind of response is expected?")
    print("    - φ^2: Explanation/Teaching")
    print("    - φ^1: Information/Facts")
    print("    - φ^0: Acknowledgment")
    print("    - φ^-1: Social response")
    print()
    print("  Dim 3: FORMALITY")
    print("    - How formal is the context?")
    print("    - φ^2: Academic/Technical")
    print("    - φ^1: Professional")
    print("    - φ^0: Casual")
    print("    - φ^-1: Informal/Friendly")
    print()
    
    # Create lattice with these dimensions
    dimensions = {
        0: 'domain',
        1: 'specificity',
        2: 'intent',
        3: 'formality',
    }
    
    lattice = PhiLattice(dimensions)
    
    # Define concepts
    concepts = {
        # Knowledge concepts
        'physics': [3, 2, 2, 2],      # High domain, specific, explanation, academic
        'science': [3, 1, 2, 1],      # High domain, general, explanation, professional
        'math': [3, 2, 2, 2],         # High domain, specific, explanation, academic
        
        # Identity concepts
        'who_are_you': [0, 0, 1, 0],  # Meta, general, information, casual
        'what_can_you_do': [0, 1, 1, 0],  # Meta, specific, information, casual
        
        # Social concepts
        'hello': [-1, -1, 0, -1],     # Social, general, acknowledgment, informal
        'thank_you': [-1, 0, 0, 0],   # Social, general, acknowledgment, casual
        'goodbye': [-1, -1, 0, -1],   # Social, general, acknowledgment, informal
    }
    
    print("Concept Positions:")
    for name, levels in concepts.items():
        pos = lattice.levels_to_position(levels)
        print(f"  {name:20s}: levels={levels}, position=[{', '.join(f'{p:.3f}' for p in pos)}]")
    print()
    
    # Test queries
    queries = {
        'what is physics': [3, 1, 2, 1],      # Asking about physics
        'who are you': [0, 0, 1, 0],          # Identity query
        'hello': [-1, -1, 0, -1],             # Greeting
        'thank you': [-1, 0, 0, 0],           # Thanks
    }
    
    print("Query Matching:")
    for query, query_levels in queries.items():
        query_pos = lattice.levels_to_position(query_levels)
        
        # Find nearest concept
        best_name = None
        best_dist = float('inf')
        for name, levels in concepts.items():
            pos = lattice.levels_to_position(levels)
            dist = lattice.distance(query_pos, pos)
            if dist < best_dist:
                best_dist = dist
                best_name = name
        
        print(f"  '{query}' → {best_name} (dist={best_dist:.3f})")


def main():
    print("=" * 70)
    print("DEEP DIVE: φ-LATTICE IMPLEMENTATION")
    print("=" * 70)
    print()
    
    lattice, concepts = test_phi_lattice()
    compare_with_eigenspace()
    explore_semantic_dimensions()
    
    print()
    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. φ-LATTICE PROVIDES ABSOLUTE COORDINATES:
   - Positions are at φ^k for integer k
   - Mathematically verifiable
   - No DC component problem

2. SEMANTIC DIMENSIONS GIVE MEANING:
   - Domain: What area of knowledge
   - Specificity: How specific
   - Intent: What kind of response
   - Formality: How formal

3. THE PROBLEM WITH EIGENSPACE:
   - Positions are compressed into [0.1, 0.5] range
   - This is only φ^(-2) to φ^(-1)
   - Very little dynamic range for discrimination
   - DC component dominates

4. THE SOLUTION:
   - Use φ-lattice for ABSOLUTE positioning
   - Define semantic dimensions with clear meaning
   - Snap concepts to lattice points
   - Navigate using φ-weighted distance

5. CONNECTION TO ZETA ZEROS:
   - Zeta zeros are at positions like 14.13, 21.02, 25.01...
   - These could be WAYPOINTS between φ-levels
   - The "natural curve" for navigation
""")


if __name__ == "__main__":
    main()
