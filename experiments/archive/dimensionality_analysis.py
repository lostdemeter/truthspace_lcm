#!/usr/bin/env python3
"""
Dimensionality Analysis
=======================

Investigating:
1. Why does the plastic constant show stronger separation than φ?
2. Is 8D optimal for our encoding? Why 12D for LLMs?
3. How can orthogonality be exploited for autotuning?

Key questions:
- What's special about the plastic constant (1.324718)?
- Do we need more or fewer dimensions?
- What determines the "right" number of axes?
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm import PhiEncoder, ClockOracle, CLOCK_RATIOS_12D, CLOCK_RATIOS_6D
from truthspace_lcm.core.encoder import BOOTSTRAP_PRIMITIVES, PrimitiveType


# =============================================================================
# MATHEMATICAL CONSTANTS ANALYSIS
# =============================================================================

# Key mathematical constants and their properties
CONSTANTS = {
    'plastic': {
        'value': 1.324717957244746,
        'equation': 'x³ = x + 1',
        'sequence': 'Padovan sequence',
        'property': 'Unique real root of cubic',
    },
    'golden': {
        'value': (1 + np.sqrt(5)) / 2,
        'equation': 'x² = x + 1',
        'sequence': 'Fibonacci sequence',
        'property': 'Unique positive root of quadratic',
    },
    'silver': {
        'value': 1 + np.sqrt(2),
        'equation': 'x² = 2x + 1',
        'sequence': 'Pell numbers',
        'property': 'Related to √2',
    },
    'bronze': {
        'value': (3 + np.sqrt(13)) / 2,
        'equation': 'x² = 3x + 1',
        'sequence': 'Generalized Fibonacci',
        'property': 'Third metallic mean',
    },
    'tribonacci': {
        'value': 1.839286755214161,
        'equation': 'x³ = x² + x + 1',
        'sequence': 'Tribonacci sequence',
        'property': 'Three-term recurrence',
    },
}


def analyze_constant_properties():
    """Analyze mathematical properties of each constant."""
    print("=" * 70)
    print("MATHEMATICAL CONSTANTS ANALYSIS")
    print("=" * 70)
    
    print("\nKey constants and their defining equations:\n")
    
    for name, props in CONSTANTS.items():
        print(f"  {name:12}: {props['value']:.6f}")
        print(f"               Equation: {props['equation']}")
        print(f"               Sequence: {props['sequence']}")
        print(f"               Property: {props['property']}")
        print()
    
    # Analyze relationships between constants
    print("-" * 70)
    print("RELATIONSHIPS BETWEEN CONSTANTS")
    print("-" * 70)
    
    phi = CONSTANTS['golden']['value']
    plastic = CONSTANTS['plastic']['value']
    silver = CONSTANTS['silver']['value']
    
    print(f"\n  φ (golden)  = {phi:.6f}")
    print(f"  ρ (plastic) = {plastic:.6f}")
    print(f"  δ (silver)  = {silver:.6f}")
    
    print(f"\n  φ² = {phi**2:.6f} = φ + 1 = {phi + 1:.6f} ✓")
    print(f"  ρ³ = {plastic**3:.6f} = ρ + 1 = {plastic + 1:.6f} ✓")
    print(f"  δ² = {silver**2:.6f} = 2δ + 1 = {2*silver + 1:.6f} ✓")
    
    # Key insight: plastic is "slower" than golden
    print(f"\n  Ratio φ/ρ = {phi/plastic:.6f}")
    print(f"  Ratio δ/φ = {silver/phi:.6f}")
    
    print("\n  💡 Insight: Plastic constant grows SLOWER than golden ratio")
    print("     This means plastic-based phases spread out more gradually,")
    print("     potentially providing finer-grained separation.")


# =============================================================================
# WHY PLASTIC SHOWS STRONGER SEPARATION
# =============================================================================

def analyze_plastic_separation():
    """Investigate why plastic constant shows stronger semantic separation."""
    print("\n" + "=" * 70)
    print("WHY PLASTIC CONSTANT SHOWS STRONGER SEPARATION")
    print("=" * 70)
    
    oracle = ClockOracle(max_n=1000)
    
    # Compare phase distributions
    print("\nPhase distribution analysis (first 100 positions):\n")
    
    for const_name in ['golden', 'plastic', 'silver']:
        phases = [oracle.get_fractional_phase(n, const_name) for n in range(1, 101)]
        
        # Analyze distribution
        hist, bins = np.histogram(phases, bins=10)
        uniformity = np.std(hist) / np.mean(hist)  # Lower = more uniform
        
        # Analyze spacing
        sorted_phases = np.sort(phases)
        gaps = np.diff(sorted_phases)
        gap_variance = np.var(gaps)
        
        print(f"  {const_name:12}:")
        print(f"    Uniformity (lower=better): {uniformity:.4f}")
        print(f"    Gap variance (lower=more even): {gap_variance:.6f}")
        print(f"    Min gap: {np.min(gaps):.4f}, Max gap: {np.max(gaps):.4f}")
        print()
    
    # Key insight about plastic
    print("-" * 70)
    print("HYPOTHESIS: Why Plastic Works Better")
    print("-" * 70)
    
    print("""
    The plastic constant (ρ ≈ 1.3247) has unique properties:
    
    1. SLOWER GROWTH: ρ < φ < δ
       - Golden ratio phases "jump" more between positions
       - Plastic phases spread more gradually
       - This provides finer discrimination between similar concepts
    
    2. CUBIC vs QUADRATIC:
       - φ satisfies x² = x + 1 (quadratic)
       - ρ satisfies x³ = x + 1 (cubic)
       - Cubic relationships create more complex interference patterns
    
    3. PADOVAN vs FIBONACCI:
       - Fibonacci: each term = sum of 2 previous
       - Padovan: each term = sum of 2nd and 3rd previous
       - Padovan has "longer memory" in its recurrence
    
    4. SEMANTIC IMPLICATION:
       - If semantic relationships have hierarchical depth > 2,
         a cubic constant might capture them better than quadratic
       - The "longer memory" of Padovan might match how meaning
         accumulates across multiple levels of abstraction
    """)


# =============================================================================
# DIMENSIONALITY ANALYSIS
# =============================================================================

def analyze_current_dimensions():
    """Analyze our current 8D encoding structure."""
    print("\n" + "=" * 70)
    print("CURRENT DIMENSIONALITY: 8D ENCODING")
    print("=" * 70)
    
    print("\nOur current primitive structure:\n")
    
    # Group primitives by dimension
    dims = defaultdict(list)
    for p in BOOTSTRAP_PRIMITIVES:
        dims[p.dimension].append(p)
    
    for dim in sorted(dims.keys()):
        primitives = dims[dim]
        print(f"  Dimension {dim}:")
        for p in primitives:
            opposite = f" (opposite: {p.opposite})" if p.opposite else ""
            print(f"    - {p.name:12} level={p.level} {p.ptype.value:8}{opposite}")
    
    # Analyze usage
    print("\n" + "-" * 70)
    print("DIMENSION USAGE ANALYSIS")
    print("-" * 70)
    
    total_primitives = len(BOOTSTRAP_PRIMITIVES)
    used_dims = len(dims)
    
    print(f"\n  Total primitives: {total_primitives}")
    print(f"  Dimensions used: {used_dims} of 8")
    print(f"  Primitives per dimension: {total_primitives / used_dims:.1f}")
    
    # Check for imbalance
    dim_counts = [len(dims[d]) for d in range(8)]
    print(f"\n  Distribution: {dim_counts}")
    print(f"  Variance: {np.var(dim_counts):.2f}")
    
    if np.var(dim_counts) > 1:
        print("  ⚠️  Uneven distribution - some dimensions underutilized")
    
    # Semantic groupings
    print("\n" + "-" * 70)
    print("SEMANTIC AXIS INTERPRETATION")
    print("-" * 70)
    
    print("""
    Current semantic axes (8D):
    
    Dim 0-3: ACTIONS (what to do)
      0: CREATE ↔ DESTROY (existence axis)
      1: READ ↔ WRITE (information flow axis)
      2: MOVE / SEARCH (spatial axis)
      3: CONNECT / EXECUTE (interaction axis)
    
    Dim 4-6: DOMAINS (what type of thing)
      4: FILE / SYSTEM
      5: PROCESS / DATA
      6: NETWORK
    
    Dim 7: MODIFIERS (how to do it)
      7: ALL / RECURSIVE / FORCE
    
    Question: Is this the RIGHT decomposition?
    """)


def analyze_optimal_dimensions():
    """Investigate what the optimal number of dimensions might be."""
    print("\n" + "=" * 70)
    print("OPTIMAL DIMENSIONALITY ANALYSIS")
    print("=" * 70)
    
    encoder = PhiEncoder()
    
    # Test concepts
    concepts = [
        "create", "destroy", "read", "write", "move", "copy",
        "file", "directory", "process", "network", "system",
        "search", "find", "list", "show", "connect", "execute",
        "compress", "archive", "delete", "remove", "recursive"
    ]
    
    # Encode all concepts
    encodings = {}
    for c in concepts:
        decomp = encoder.encode(c)
        encodings[c] = decomp.position
    
    # Analyze which dimensions are actually used
    all_positions = np.array(list(encodings.values()))
    
    print("\nDimension activation analysis:\n")
    
    for dim in range(8):
        dim_values = all_positions[:, dim]
        non_zero = np.sum(np.abs(dim_values) > 0.01)
        variance = np.var(dim_values)
        
        status = "✓ ACTIVE" if non_zero > 2 else "⚠️ SPARSE" if non_zero > 0 else "✗ UNUSED"
        print(f"  Dim {dim}: {non_zero:2d}/{len(concepts)} concepts, var={variance:.4f} {status}")
    
    # Compute effective dimensionality via PCA
    print("\n" + "-" * 70)
    print("EFFECTIVE DIMENSIONALITY (PCA)")
    print("-" * 70)
    
    # Center the data
    centered = all_positions - np.mean(all_positions, axis=0)
    
    # SVD to get singular values
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    # Explained variance ratio
    explained_var = (S ** 2) / np.sum(S ** 2)
    cumulative_var = np.cumsum(explained_var)
    
    print("\nSingular value analysis:\n")
    for i, (s, ev, cv) in enumerate(zip(S, explained_var, cumulative_var)):
        bar = "█" * int(ev * 50)
        print(f"  Dim {i}: σ={s:.4f}, var={ev:.2%}, cumulative={cv:.2%} {bar}")
    
    # Find effective dimensionality (95% variance)
    effective_dim = np.argmax(cumulative_var >= 0.95) + 1
    print(f"\n  Effective dimensionality (95% variance): {effective_dim}")
    
    # Find intrinsic dimensionality (elbow)
    diffs = np.diff(S)
    elbow = np.argmax(np.abs(diffs)) + 1
    print(f"  Intrinsic dimensionality (elbow): {elbow}")
    
    return effective_dim, elbow


def analyze_llm_dimensions():
    """Discuss why LLMs use certain dimensions."""
    print("\n" + "=" * 70)
    print("WHY 12D FOR THE CLOCK? WHY 768D/4096D FOR LLMS?")
    print("=" * 70)
    
    print("""
    LLM EMBEDDING DIMENSIONS:
    
    Common sizes:
    - GPT-2 small:  768D
    - GPT-2 medium: 1024D
    - GPT-3:        12288D
    - BERT base:    768D
    - BERT large:   1024D
    
    Why these numbers?
    - 768 = 12 × 64 (12 attention heads × 64 dim per head)
    - 1024 = 16 × 64
    - Powers of 2 for computational efficiency
    
    The 12 ATTENTION HEADS are interesting:
    - Each head learns different relationship types
    - 12 seems to be a "sweet spot" for capturing diverse patterns
    - Our 12D clock mirrors this structure!
    
    THE 12D CLOCK RATIOS:
    - 6 "metallic means" (golden, silver, bronze, etc.)
    - 6 additional constants (plastic, tribonacci, etc.)
    - Each creates different interference patterns
    - Together they span a rich space of phase relationships
    
    HYPOTHESIS: 12 dimensions capture fundamental relationship types:
    1. Hierarchical (parent-child)
    2. Sequential (before-after)
    3. Causal (cause-effect)
    4. Compositional (part-whole)
    5. Oppositional (antonyms)
    6. Synonymic (same meaning)
    7. Analogical (A:B :: C:D)
    8. Associative (co-occurrence)
    9. Functional (same role)
    10. Categorical (same type)
    11. Spatial (location-based)
    12. Temporal (time-based)
    """)
    
    print("-" * 70)
    print("OUR 8D vs 12D CLOCK")
    print("-" * 70)
    
    print("""
    Current situation:
    - Our φ-encoder uses 8D
    - The clock uses 12D
    - We're only using first 8 clock dimensions for modulation
    
    Questions:
    1. Should we expand to 12D encoding?
    2. Are we missing relationship types with only 8D?
    3. What would dimensions 9-12 represent semantically?
    
    Possible expansion:
    - Dim 8: TEMPORAL (before, after, during, while)
    - Dim 9: CAUSAL (because, therefore, causes, results)
    - Dim 10: CONDITIONAL (if, when, unless, provided)
    - Dim 11: COMPARATIVE (more, less, equal, different)
    """)


# =============================================================================
# ORTHOGONALITY EXPLOITATION
# =============================================================================

def analyze_orthogonality():
    """Explore how orthogonality can be exploited for autotuning."""
    print("\n" + "=" * 70)
    print("ORTHOGONALITY EXPLOITATION FOR AUTOTUNING")
    print("=" * 70)
    
    print("""
    KEY INSIGHT: Orthogonal dimensions are INDEPENDENT
    
    If concept A is on dimension 0 and concept B is on dimension 4,
    they have similarity = 0 (completely independent).
    
    This means:
    1. Adding knowledge to dim 0 CANNOT interfere with dim 4
    2. We can tune dimensions independently
    3. Collisions only happen WITHIN a dimension
    
    AUTOTUNING STRATEGY:
    
    Step 1: CLASSIFY the new concept
      - What type is it? (ACTION, DOMAIN, MODIFIER)
      - This determines which dimensions it can occupy
    
    Step 2: FIND THE RIGHT DIMENSION
      - Within its type, which specific dimension?
      - Check for collisions only in that dimension
    
    Step 3: FIND THE RIGHT LEVEL
      - Within the dimension, what φ^level?
      - Opposites get opposite signs
      - Synonyms get same position
    
    Step 4: VERIFY ORTHOGONALITY
      - Confirm new concept is orthogonal to unrelated concepts
      - If not, we've misclassified it
    """)
    
    # Demonstrate with example
    print("-" * 70)
    print("EXAMPLE: Adding 'backup' concept")
    print("-" * 70)
    
    encoder = PhiEncoder()
    
    # What is 'backup' related to?
    related = ["copy", "move", "archive", "save"]
    unrelated = ["delete", "network", "process", "search"]
    
    print("\n  Analyzing 'backup'...")
    print("\n  Related concepts:")
    for c in related:
        decomp = encoder.encode(c)
        print(f"    {c:12}: position = {decomp.position[:4]}...")
    
    print("\n  Unrelated concepts:")
    for c in unrelated:
        decomp = encoder.encode(c)
        print(f"    {c:12}: position = {decomp.position[:4]}...")
    
    # Determine where backup should go
    copy_pos = encoder.encode("copy").position
    move_pos = encoder.encode("move").position
    
    print(f"\n  'copy' is on dimension {np.argmax(np.abs(copy_pos))}")
    print(f"  'move' is on dimension {np.argmax(np.abs(move_pos))}")
    print(f"\n  → 'backup' should go on dimension 2 (MOVE/COPY axis)")
    print(f"  → Level should be similar to 'copy' (same polarity)")
    
    print("""
    
    AUTOTUNER ALGORITHM:
    
    def find_optimal_position(new_concept, test_cases):
        # 1. Encode test case inputs to find semantic neighborhood
        neighbors = [encode(tc.input) for tc in test_cases]
        
        # 2. Find which dimension(s) neighbors occupy
        active_dims = find_active_dimensions(neighbors)
        
        # 3. For each candidate dimension, check for collisions
        for dim in active_dims:
            existing = get_concepts_on_dimension(dim)
            if no_collision(new_concept, existing):
                return dim, compute_level(new_concept, neighbors)
        
        # 4. If all dimensions have collisions, we need a new dimension
        return suggest_new_dimension()
    """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all dimensionality analyses."""
    
    # Part 1: Why plastic constant?
    analyze_constant_properties()
    analyze_plastic_separation()
    
    # Part 2: Dimensionality
    analyze_current_dimensions()
    effective_dim, intrinsic_dim = analyze_optimal_dimensions()
    analyze_llm_dimensions()
    
    # Part 3: Orthogonality exploitation
    analyze_orthogonality()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    print(f"""
    FINDINGS:
    
    1. PLASTIC CONSTANT
       - Shows stronger separation because it grows slower than φ
       - Cubic recurrence captures deeper hierarchical structure
       - Consider using plastic as primary constant, φ as secondary
    
    2. DIMENSIONALITY
       - Current: 8D encoding, 12D clock
       - Effective: ~{effective_dim}D (based on variance)
       - Intrinsic: ~{intrinsic_dim}D (based on structure)
       - Recommendation: Consider expanding to 12D for full clock alignment
    
    3. ORTHOGONALITY
       - Key insight: independent dimensions = independent tuning
       - Autotuner should classify → find dimension → find level
       - Collisions only matter within a dimension
    
    NEXT STEPS:
    
    1. Experiment with plastic-primary encoding
    2. Add dimensions 8-11 for temporal/causal/conditional/comparative
    3. Implement dimension-aware autotuning
    4. Test if 12D encoding improves semantic separation
    """)


if __name__ == "__main__":
    main()
