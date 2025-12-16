#!/usr/bin/env python3
"""
Vacuum Forming Hypothesis - Experimental Tests
===============================================

Testing whether there's deeper internal structure in semantic space
that training only captures the surface of.

Experiments:
1. Phase-shift consistency - Do related concepts stay related across phases?
2. φ-geometry alignment - Does our encoding capture semantic relationships?
3. Structure discovery - Can we find patterns that suggest interior structure?
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm import TruthSpace, PhiEncoder, ClockOracle, CLOCK_RATIOS_12D
from truthspace_lcm.core.truthspace import EntryType


# =============================================================================
# TEST DATA: Semantic Concept Pairs
# =============================================================================

# Pairs of concepts with known relationships
RELATED_PAIRS = [
    # Hierarchical (parent-child)
    ("file", "directory"),
    ("process", "system"),
    ("read", "write"),
    ("create", "destroy"),
    
    # Functional (do similar things)
    ("copy", "move"),
    ("compress", "archive"),
    ("search", "find"),
    ("list", "show"),
    
    # Domain (same category)
    ("ssh", "network"),
    ("grep", "search"),
    ("tar", "compress"),
    ("chmod", "permissions"),
]

UNRELATED_PAIRS = [
    # Semantically distant
    ("file", "network"),
    ("compress", "ssh"),
    ("create", "search"),
    ("process", "directory"),
    
    # Different domains entirely
    ("read", "compress"),
    ("list", "connect"),
    ("move", "execute"),
    ("write", "find"),
]


# =============================================================================
# EXPERIMENT 1: Phase-Shift Consistency
# =============================================================================

@dataclass
class ConsistencyResult:
    """Result of phase-shift consistency test."""
    pair: Tuple[str, str]
    is_related: bool
    similarities: List[float]
    mean_similarity: float
    variance: float
    min_similarity: float
    max_similarity: float


def experiment_1_phase_consistency(
    oracle: ClockOracle,
    encoder: PhiEncoder,
    n_phases: int = 100,
) -> Dict:
    """
    Test: Do related concepts maintain consistent similarity across phase shifts?
    
    Hypothesis: Related concepts should have LOW variance (consistently similar).
    Unrelated concepts should have HIGH variance (similarity fluctuates).
    """
    print("=" * 70)
    print("EXPERIMENT 1: Phase-Shift Consistency")
    print("=" * 70)
    print(f"\nTesting {len(RELATED_PAIRS)} related pairs and {len(UNRELATED_PAIRS)} unrelated pairs")
    print(f"Across {n_phases} phase shifts\n")
    
    results = {
        "related": [],
        "unrelated": [],
    }
    
    def test_pair(concept1: str, concept2: str, is_related: bool) -> ConsistencyResult:
        """Test a single concept pair across phase shifts."""
        # Get base encodings (extract position vector from SemanticDecomposition)
        decomp1 = encoder.encode(concept1)
        decomp2 = encoder.encode(concept2)
        v1_base = decomp1.position
        v2_base = decomp2.position
        
        similarities = []
        encoding_dim = len(v1_base)  # Get actual encoding dimension (8D)
        
        for phase_n in range(1, n_phases + 1):
            # Get phase vector
            phase_vec = oracle.get_12d_phase(phase_n * 10)  # Sample every 10th position
            
            # Apply phase modulation to encodings
            # We'll use the phase to rotate/modulate the similarity computation
            # This simulates "viewing" the relationship from different angles
            
            # Method: Modulate each dimension by corresponding phase
            # Use first 8 dimensions of 12D phase vector to match encoding
            phase_mod = phase_vec[:encoding_dim]
            
            # Apply phase rotation
            v1_shifted = v1_base * np.cos(2 * np.pi * phase_mod)
            v2_shifted = v2_base * np.cos(2 * np.pi * phase_mod)
            
            # Compute similarity
            norm1 = np.linalg.norm(v1_shifted)
            norm2 = np.linalg.norm(v2_shifted)
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(v1_shifted, v2_shifted) / (norm1 * norm2)
            else:
                sim = 0.0
            
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        return ConsistencyResult(
            pair=(concept1, concept2),
            is_related=is_related,
            similarities=similarities.tolist(),
            mean_similarity=float(np.mean(similarities)),
            variance=float(np.var(similarities)),
            min_similarity=float(np.min(similarities)),
            max_similarity=float(np.max(similarities)),
        )
    
    # Test related pairs
    print("Testing related pairs...")
    for c1, c2 in RELATED_PAIRS:
        result = test_pair(c1, c2, is_related=True)
        results["related"].append(result)
        print(f"  {c1:12} ↔ {c2:12}: mean={result.mean_similarity:+.3f}, var={result.variance:.4f}")
    
    print("\nTesting unrelated pairs...")
    for c1, c2 in UNRELATED_PAIRS:
        result = test_pair(c1, c2, is_related=False)
        results["unrelated"].append(result)
        print(f"  {c1:12} ↔ {c2:12}: mean={result.mean_similarity:+.3f}, var={result.variance:.4f}")
    
    # Analyze results
    related_variances = [r.variance for r in results["related"]]
    unrelated_variances = [r.variance for r in results["unrelated"]]
    
    related_means = [r.mean_similarity for r in results["related"]]
    unrelated_means = [r.mean_similarity for r in results["unrelated"]]
    
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)
    
    print(f"\nRelated pairs:")
    print(f"  Mean similarity: {np.mean(related_means):.4f} (±{np.std(related_means):.4f})")
    print(f"  Mean variance:   {np.mean(related_variances):.6f}")
    
    print(f"\nUnrelated pairs:")
    print(f"  Mean similarity: {np.mean(unrelated_means):.4f} (±{np.std(unrelated_means):.4f})")
    print(f"  Mean variance:   {np.mean(unrelated_variances):.6f}")
    
    # Hypothesis test
    print("\n" + "-" * 70)
    print("HYPOTHESIS TEST")
    print("-" * 70)
    
    # H0: Related and unrelated pairs have same variance
    # H1: Related pairs have lower variance (more consistent)
    
    variance_ratio = np.mean(related_variances) / (np.mean(unrelated_variances) + 1e-10)
    
    print(f"\nVariance ratio (related/unrelated): {variance_ratio:.4f}")
    
    if variance_ratio < 0.8:
        print("✅ SUPPORTED: Related concepts show more consistent similarity across phases")
        hypothesis_supported = True
    elif variance_ratio > 1.2:
        print("❌ REFUTED: Related concepts show LESS consistent similarity (unexpected)")
        hypothesis_supported = False
    else:
        print("⚠️  INCONCLUSIVE: No significant difference in variance")
        hypothesis_supported = None
    
    return {
        "results": results,
        "related_mean_variance": np.mean(related_variances),
        "unrelated_mean_variance": np.mean(unrelated_variances),
        "variance_ratio": variance_ratio,
        "hypothesis_supported": hypothesis_supported,
    }


# =============================================================================
# EXPERIMENT 2: φ-Geometry Semantic Alignment
# =============================================================================

def experiment_2_phi_alignment(
    ts: TruthSpace,
    encoder: PhiEncoder,
) -> Dict:
    """
    Test: Does our φ-encoding capture semantic relationships?
    
    Compare distances in φ-space to semantic relationships.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: φ-Geometry Semantic Alignment")
    print("=" * 70)
    
    # Get all concepts from TruthSpace
    all_pairs = RELATED_PAIRS + UNRELATED_PAIRS
    concepts = list(set([c for pair in all_pairs for c in pair]))
    
    print(f"\nAnalyzing {len(concepts)} concepts in φ-space\n")
    
    # Encode all concepts (extract position vectors)
    encodings = {}
    for concept in concepts:
        decomp = encoder.encode(concept)
        encodings[concept] = decomp.position
    
    # Compute pairwise distances
    def phi_distance(c1: str, c2: str) -> float:
        v1, v2 = encodings[c1], encodings[c2]
        return float(np.linalg.norm(v1 - v2))
    
    def phi_similarity(c1: str, c2: str) -> float:
        v1, v2 = encodings[c1], encodings[c2]
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 > 0 and norm2 > 0:
            return float(np.dot(v1, v2) / (norm1 * norm2))
        return 0.0
    
    # Compare related vs unrelated
    related_distances = [phi_distance(c1, c2) for c1, c2 in RELATED_PAIRS]
    unrelated_distances = [phi_distance(c1, c2) for c1, c2 in UNRELATED_PAIRS]
    
    related_similarities = [phi_similarity(c1, c2) for c1, c2 in RELATED_PAIRS]
    unrelated_similarities = [phi_similarity(c1, c2) for c1, c2 in UNRELATED_PAIRS]
    
    print("φ-Space Distances:")
    print(f"  Related pairs:   mean={np.mean(related_distances):.4f} (±{np.std(related_distances):.4f})")
    print(f"  Unrelated pairs: mean={np.mean(unrelated_distances):.4f} (±{np.std(unrelated_distances):.4f})")
    
    print("\nφ-Space Similarities:")
    print(f"  Related pairs:   mean={np.mean(related_similarities):.4f} (±{np.std(related_similarities):.4f})")
    print(f"  Unrelated pairs: mean={np.mean(unrelated_similarities):.4f} (±{np.std(unrelated_similarities):.4f})")
    
    # Check if related pairs are closer in φ-space
    distance_ratio = np.mean(related_distances) / (np.mean(unrelated_distances) + 1e-10)
    similarity_diff = np.mean(related_similarities) - np.mean(unrelated_similarities)
    
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)
    
    print(f"\nDistance ratio (related/unrelated): {distance_ratio:.4f}")
    print(f"Similarity difference (related - unrelated): {similarity_diff:+.4f}")
    
    if distance_ratio < 0.9 and similarity_diff > 0.05:
        print("\n✅ ALIGNED: φ-geometry captures semantic relationships")
        print("   Related concepts are closer in φ-space")
        aligned = True
    elif distance_ratio > 1.1 or similarity_diff < -0.05:
        print("\n❌ MISALIGNED: φ-geometry does NOT capture semantic relationships")
        aligned = False
    else:
        print("\n⚠️  WEAK ALIGNMENT: Some structure captured, but not strong")
        aligned = None
    
    # Detailed breakdown
    print("\n" + "-" * 70)
    print("DETAILED PAIR ANALYSIS")
    print("-" * 70)
    
    print("\nRelated pairs (should be similar):")
    for (c1, c2), sim in zip(RELATED_PAIRS, related_similarities):
        marker = "✓" if sim > 0.5 else "✗"
        print(f"  {marker} {c1:12} ↔ {c2:12}: sim={sim:+.4f}")
    
    print("\nUnrelated pairs (should be dissimilar):")
    for (c1, c2), sim in zip(UNRELATED_PAIRS, unrelated_similarities):
        marker = "✓" if sim < 0.5 else "✗"
        print(f"  {marker} {c1:12} ↔ {c2:12}: sim={sim:+.4f}")
    
    return {
        "distance_ratio": distance_ratio,
        "similarity_diff": similarity_diff,
        "aligned": aligned,
        "related_similarities": related_similarities,
        "unrelated_similarities": unrelated_similarities,
    }


# =============================================================================
# EXPERIMENT 3: Structure Discovery
# =============================================================================

def experiment_3_structure_discovery(
    oracle: ClockOracle,
    encoder: PhiEncoder,
) -> Dict:
    """
    Test: Can we discover deeper structure by analyzing phase patterns?
    
    Look for:
    - Resonance patterns (certain phases reveal clearer structure)
    - Dimensional correlations (which clock dimensions matter most)
    - Hidden clusters (groups that emerge across phases)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Structure Discovery")
    print("=" * 70)
    
    concepts = list(set([c for pair in RELATED_PAIRS + UNRELATED_PAIRS for c in pair]))
    
    print(f"\nSearching for hidden structure in {len(concepts)} concepts...")
    print(f"Using {len(CLOCK_RATIOS_12D)} clock dimensions\n")
    
    # Analyze each clock dimension separately
    print("-" * 70)
    print("DIMENSIONAL ANALYSIS: Which clock ratios reveal most structure?")
    print("-" * 70)
    
    dimension_scores = {}
    
    for dim_name in CLOCK_RATIOS_12D.keys():
        # For this dimension, compute how well it separates related from unrelated
        related_sims = []
        unrelated_sims = []
        
        for c1, c2 in RELATED_PAIRS:
            v1 = encoder.encode(c1).position
            v2 = encoder.encode(c2).position
            
            # Get phase for this dimension at different positions
            phase1 = oracle.get_fractional_phase(hash(c1) % 1000, dim_name)
            phase2 = oracle.get_fractional_phase(hash(c2) % 1000, dim_name)
            
            # Phase-modulated similarity
            mod = np.cos(2 * np.pi * (phase1 - phase2))
            base_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            related_sims.append(base_sim * mod)
        
        for c1, c2 in UNRELATED_PAIRS:
            v1 = encoder.encode(c1).position
            v2 = encoder.encode(c2).position
            
            phase1 = oracle.get_fractional_phase(hash(c1) % 1000, dim_name)
            phase2 = oracle.get_fractional_phase(hash(c2) % 1000, dim_name)
            
            mod = np.cos(2 * np.pi * (phase1 - phase2))
            base_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            unrelated_sims.append(base_sim * mod)
        
        # Score = how well this dimension separates related from unrelated
        separation = np.mean(related_sims) - np.mean(unrelated_sims)
        dimension_scores[dim_name] = separation
    
    # Sort by separation power
    sorted_dims = sorted(dimension_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\nClock dimensions ranked by semantic separation power:\n")
    for dim_name, score in sorted_dims:
        ratio = CLOCK_RATIOS_12D[dim_name]
        bar = "█" * int(abs(score) * 50)
        direction = "+" if score > 0 else "-"
        print(f"  {dim_name:12} (ratio={ratio:.4f}): {direction}{abs(score):.4f} {bar}")
    
    # Find the best dimension
    best_dim = sorted_dims[0][0]
    best_score = sorted_dims[0][1]
    
    print(f"\n🔍 Best dimension: {best_dim} (ratio={CLOCK_RATIOS_12D[best_dim]:.6f})")
    
    # Analyze resonance patterns
    print("\n" + "-" * 70)
    print("RESONANCE ANALYSIS: Do certain phase positions reveal clearer structure?")
    print("-" * 70)
    
    # Test different phase positions
    phase_scores = []
    
    for phase_n in range(1, 101):
        phase_vec = oracle.get_12d_phase(phase_n * 10)
        
        # Compute separation at this phase
        related_sims = []
        unrelated_sims = []
        
        for c1, c2 in RELATED_PAIRS:
            v1 = encoder.encode(c1).position
            v2 = encoder.encode(c2).position
            
            # Modulate by full phase vector (use first 8 dims to match encoding)
            v1_mod = v1 * np.cos(2 * np.pi * phase_vec[:len(v1)])
            v2_mod = v2 * np.cos(2 * np.pi * phase_vec[:len(v2)])
            
            sim = np.dot(v1_mod, v2_mod) / (np.linalg.norm(v1_mod) * np.linalg.norm(v2_mod) + 1e-10)
            related_sims.append(sim)
        
        for c1, c2 in UNRELATED_PAIRS:
            v1 = encoder.encode(c1).position
            v2 = encoder.encode(c2).position
            
            v1_mod = v1 * np.cos(2 * np.pi * phase_vec[:len(v1)])
            v2_mod = v2 * np.cos(2 * np.pi * phase_vec[:len(v2)])
            
            sim = np.dot(v1_mod, v2_mod) / (np.linalg.norm(v1_mod) * np.linalg.norm(v2_mod) + 1e-10)
            unrelated_sims.append(sim)
        
        separation = np.mean(related_sims) - np.mean(unrelated_sims)
        phase_scores.append((phase_n * 10, separation))
    
    # Find resonance peaks
    scores = [s for _, s in phase_scores]
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    peaks = [(n, s) for n, s in phase_scores if s > mean_score + std_score]
    troughs = [(n, s) for n, s in phase_scores if s < mean_score - std_score]
    
    print(f"\nMean separation: {mean_score:.4f} (±{std_score:.4f})")
    print(f"Found {len(peaks)} resonance peaks and {len(troughs)} troughs")
    
    if peaks:
        print("\n🔺 Resonance peaks (clearer structure):")
        for n, s in sorted(peaks, key=lambda x: x[1], reverse=True)[:5]:
            print(f"   Phase n={n}: separation={s:.4f}")
    
    if troughs:
        print("\n🔻 Resonance troughs (muddier structure):")
        for n, s in sorted(troughs, key=lambda x: x[1])[:5]:
            print(f"   Phase n={n}: separation={s:.4f}")
    
    # Look for periodicity in resonance
    print("\n" + "-" * 70)
    print("PERIODICITY ANALYSIS: Is there a pattern to the resonances?")
    print("-" * 70)
    
    # FFT of separation scores
    fft = np.fft.fft(scores)
    freqs = np.fft.fftfreq(len(scores))
    
    # Find dominant frequencies
    magnitudes = np.abs(fft)
    dominant_idx = np.argsort(magnitudes)[-5:][::-1]
    
    print("\nDominant frequencies in resonance pattern:")
    for idx in dominant_idx[1:]:  # Skip DC component
        freq = freqs[idx]
        mag = magnitudes[idx]
        if freq > 0:
            period = 1 / freq
            print(f"  Frequency={freq:.4f}, Period≈{period:.1f} phases, Magnitude={mag:.2f}")
    
    # Check if any period matches φ-related values
    phi_periods = [1.618, 2.618, 4.236, 6.854]  # φ, φ², φ³, φ⁴
    
    print("\n🔍 Checking for φ-related periodicities...")
    for idx in dominant_idx[1:]:
        freq = freqs[idx]
        if freq > 0:
            period = 1 / freq
            for phi_p in phi_periods:
                if abs(period - phi_p * 10) < 5:  # Within 5 phases
                    print(f"  ✨ Period {period:.1f} ≈ {phi_p:.3f} × 10 (φ-related!)")
    
    return {
        "dimension_scores": dimension_scores,
        "best_dimension": best_dim,
        "phase_scores": phase_scores,
        "peaks": peaks,
        "troughs": troughs,
    }


# =============================================================================
# MAIN
# =============================================================================

def run_all_experiments():
    """Run all hypothesis tests."""
    print("\n" + "=" * 70)
    print("VACUUM FORMING HYPOTHESIS - EXPERIMENTAL TESTS")
    print("=" * 70)
    print("\nHypothesis: LLMs learn the 'surface' of semantic structure.")
    print("We're looking for evidence of deeper 'interior' structure.\n")
    
    # Initialize
    ts = TruthSpace()
    encoder = PhiEncoder()
    oracle = ClockOracle(max_n=10000)
    
    results = {}
    
    # Run experiments
    results["exp1"] = experiment_1_phase_consistency(oracle, encoder)
    results["exp2"] = experiment_2_phi_alignment(ts, encoder)
    results["exp3"] = experiment_3_structure_discovery(oracle, encoder)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nExperiment 1 (Phase Consistency):")
    if results["exp1"]["hypothesis_supported"] is True:
        print("  ✅ Related concepts show consistent similarity across phases")
    elif results["exp1"]["hypothesis_supported"] is False:
        print("  ❌ No consistency found")
    else:
        print("  ⚠️  Inconclusive")
    
    print("\nExperiment 2 (φ-Alignment):")
    if results["exp2"]["aligned"] is True:
        print("  ✅ φ-geometry captures semantic relationships")
    elif results["exp2"]["aligned"] is False:
        print("  ❌ φ-geometry does not align with semantics")
    else:
        print("  ⚠️  Weak alignment")
    
    print("\nExperiment 3 (Structure Discovery):")
    best_dim = results["exp3"]["best_dimension"]
    print(f"  🔍 Best clock dimension: {best_dim}")
    print(f"  🔍 Found {len(results['exp3']['peaks'])} resonance peaks")
    
    # Overall assessment
    print("\n" + "-" * 70)
    print("OVERALL ASSESSMENT")
    print("-" * 70)
    
    evidence_for = 0
    evidence_against = 0
    
    if results["exp1"]["hypothesis_supported"] is True:
        evidence_for += 1
    elif results["exp1"]["hypothesis_supported"] is False:
        evidence_against += 1
    
    if results["exp2"]["aligned"] is True:
        evidence_for += 1
    elif results["exp2"]["aligned"] is False:
        evidence_against += 1
    
    if len(results["exp3"]["peaks"]) > 5:
        evidence_for += 1
        print("\n  Resonance patterns suggest structured interior")
    
    print(f"\n  Evidence FOR interior structure: {evidence_for}")
    print(f"  Evidence AGAINST: {evidence_against}")
    
    if evidence_for > evidence_against:
        print("\n  📊 PRELIMINARY CONCLUSION: Evidence supports the vacuum forming hypothesis")
        print("     There appears to be structure beyond what simple correlation captures.")
    elif evidence_against > evidence_for:
        print("\n  📊 PRELIMINARY CONCLUSION: Evidence does not support the hypothesis")
        print("     The surface may be all there is.")
    else:
        print("\n  📊 PRELIMINARY CONCLUSION: Inconclusive - more experiments needed")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_all_experiments()
