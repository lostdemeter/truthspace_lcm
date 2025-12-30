#!/usr/bin/env python3
"""
Position Worthiness: Why Do Some Concepts Get Single-Word Names?

Key Question: What makes a position in truthspace "worthy" of a single-word name?

Hypothesis: Positions that get named are those that:
1. Lie at INTERSECTIONS of multiple self-similar transformations
2. Are FREQUENTLY VISITED in conceptual navigation (Zipf connection)
3. Represent STABLE ATTRACTORS in the geometric structure
4. Have HIGH CONNECTIVITY to other named positions

The Connection to Zipf's Law:
- Zipf: frequency ∝ 1/rank
- In truthspace: "frequency" might be how often a position is REACHED
- Positions that are reached by many paths get named
- Positions that are reached by few paths stay unnamed (lexical gaps)

Author: Lesley Gushurst
License: GPLv3
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.semantic_quaternion import SemanticQuaternion, DEFAULT_SEMANTIC_FEATURES
from experiments.self_similar_truthspace import SelfSimilarTruthSpace, Transformation


@dataclass
class PositionAnalysis:
    """Analysis of a position's worthiness for naming."""
    quaternion: SemanticQuaternion
    names: List[str]
    
    # Worthiness metrics
    reachability: int = 0  # How many transformations reach this position
    connectivity: int = 0  # How many named positions are 1 transform away
    centrality: float = 0.0  # Distance from origin (0,0,0,0)
    symmetry_score: float = 0.0  # Does symmetric position exist?
    intersection_count: int = 0  # How many transformation axes intersect here
    
    @property
    def is_named(self) -> bool:
        return len(self.names) > 0
    
    @property
    def worthiness_score(self) -> float:
        """Combined worthiness score."""
        return (
            self.reachability * 2.0 +
            self.connectivity * 1.5 +
            self.symmetry_score * 1.0 +
            self.intersection_count * 2.0 -
            self.centrality * 0.5  # Penalty for being far from center
        )


class PositionWorthinessAnalyzer:
    """
    Analyze what makes positions worthy of single-word names.
    """
    
    def __init__(self):
        self.ts = SelfSimilarTruthSpace()
        self.analyses: Dict[Tuple[float, float, float, float], PositionAnalysis] = {}
        
        # Build analyses for all known positions
        self._analyze_all_positions()
    
    def _quantize(self, sq: SemanticQuaternion, resolution: float = 0.5) -> Tuple[float, float, float, float]:
        """Quantize a quaternion to a grid position."""
        return (
            round(sq.x / resolution) * resolution,
            round(sq.y / resolution) * resolution,
            round(sq.z / resolution) * resolution,
            round(sq.w / resolution) * resolution,
        )
    
    def _analyze_all_positions(self):
        """Analyze all positions in the space."""
        # First, create analyses for all named positions
        for name, sq in DEFAULT_SEMANTIC_FEATURES.items():
            key = self._quantize(sq)
            if key not in self.analyses:
                self.analyses[key] = PositionAnalysis(
                    quaternion=sq,
                    names=[name],
                )
            else:
                self.analyses[key].names.append(name)
        
        # Calculate metrics for each position
        for key, analysis in self.analyses.items():
            self._calculate_reachability(key, analysis)
            self._calculate_connectivity(key, analysis)
            self._calculate_centrality(key, analysis)
            self._calculate_symmetry(key, analysis)
            self._calculate_intersections(key, analysis)
    
    def _calculate_reachability(self, key: Tuple, analysis: PositionAnalysis):
        """How many transformations can reach this position from other named positions?"""
        count = 0
        for source_name, source_sq in DEFAULT_SEMANTIC_FEATURES.items():
            if source_name in analysis.names:
                continue
            for transform in self.ts.transformations.values():
                result = transform.apply(source_sq)
                result_key = self._quantize(result)
                if result_key == key:
                    count += 1
        analysis.reachability = count
    
    def _calculate_connectivity(self, key: Tuple, analysis: PositionAnalysis):
        """How many named positions are exactly 1 transformation away?"""
        count = 0
        for transform in self.ts.transformations.values():
            # Apply transform to this position
            result = transform.apply(analysis.quaternion)
            result_key = self._quantize(result)
            if result_key in self.analyses and result_key != key:
                if self.analyses[result_key].is_named:
                    count += 1
        analysis.connectivity = count
    
    def _calculate_centrality(self, key: Tuple, analysis: PositionAnalysis):
        """Distance from the origin (neutral position)."""
        q = analysis.quaternion
        analysis.centrality = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
    
    def _calculate_symmetry(self, key: Tuple, analysis: PositionAnalysis):
        """Does a symmetric position (gender-flipped) exist and is it named?"""
        q = analysis.quaternion
        # Check gender-symmetric position
        symmetric_key = self._quantize(SemanticQuaternion(-q.x, q.y, q.z, q.w))
        if symmetric_key in self.analyses and self.analyses[symmetric_key].is_named:
            analysis.symmetry_score = 1.0
        else:
            analysis.symmetry_score = 0.0
    
    def _calculate_intersections(self, key: Tuple, analysis: PositionAnalysis):
        """How many transformation axes pass through this position?"""
        # A position is an "intersection" if it can be reached by multiple
        # DIFFERENT types of transformations
        reaching_transforms = set()
        for source_name, source_sq in DEFAULT_SEMANTIC_FEATURES.items():
            if source_name in analysis.names:
                continue
            for t_name, transform in self.ts.transformations.items():
                result = transform.apply(source_sq)
                result_key = self._quantize(result)
                if result_key == key:
                    # Get base transform name (without inverse_)
                    base_name = t_name.replace('inverse_', '')
                    reaching_transforms.add(base_name)
        analysis.intersection_count = len(reaching_transforms)
    
    def get_worthiness_ranking(self) -> List[Tuple[PositionAnalysis, float]]:
        """Get all positions ranked by worthiness."""
        ranked = []
        for analysis in self.analyses.values():
            ranked.append((analysis, analysis.worthiness_score))
        ranked.sort(key=lambda x: -x[1])
        return ranked
    
    def compare_named_vs_unnamed(self) -> Dict:
        """Compare metrics between named and unnamed positions."""
        named_metrics = {
            'reachability': [],
            'connectivity': [],
            'centrality': [],
            'symmetry': [],
            'intersections': [],
            'worthiness': [],
        }
        unnamed_metrics = {
            'reachability': [],
            'connectivity': [],
            'centrality': [],
            'symmetry': [],
            'intersections': [],
            'worthiness': [],
        }
        
        # Also analyze gap positions
        gaps = self.ts.find_all_predictions()
        for gap in gaps:
            key = self._quantize(gap.quaternion)
            if key not in self.analyses:
                self.analyses[key] = PositionAnalysis(
                    quaternion=gap.quaternion,
                    names=[],
                )
                # Calculate metrics for gap
                analysis = self.analyses[key]
                self._calculate_reachability(key, analysis)
                self._calculate_connectivity(key, analysis)
                self._calculate_centrality(key, analysis)
                self._calculate_symmetry(key, analysis)
                self._calculate_intersections(key, analysis)
        
        for analysis in self.analyses.values():
            target = named_metrics if analysis.is_named else unnamed_metrics
            target['reachability'].append(analysis.reachability)
            target['connectivity'].append(analysis.connectivity)
            target['centrality'].append(analysis.centrality)
            target['symmetry'].append(analysis.symmetry_score)
            target['intersections'].append(analysis.intersection_count)
            target['worthiness'].append(analysis.worthiness_score)
        
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0
        
        return {
            'named': {k: avg(v) for k, v in named_metrics.items()},
            'unnamed': {k: avg(v) for k, v in unnamed_metrics.items()},
            'named_count': len([a for a in self.analyses.values() if a.is_named]),
            'unnamed_count': len([a for a in self.analyses.values() if not a.is_named]),
        }
    
    def find_naming_threshold(self) -> float:
        """Find the worthiness threshold that separates named from unnamed."""
        named_scores = []
        unnamed_scores = []
        
        for analysis in self.analyses.values():
            if analysis.is_named:
                named_scores.append(analysis.worthiness_score)
            else:
                unnamed_scores.append(analysis.worthiness_score)
        
        if not named_scores or not unnamed_scores:
            return 0.0
        
        # Find the threshold that best separates them
        min_named = min(named_scores)
        max_unnamed = max(unnamed_scores)
        
        return (min_named + max_unnamed) / 2


def demo():
    """Demonstrate position worthiness analysis."""
    print("=" * 70)
    print("POSITION WORTHINESS: WHY DO SOME CONCEPTS GET NAMES?")
    print("=" * 70)
    print("""
    Hypothesis: Positions that get single-word names are those that:
    1. Are REACHABLE by many transformations from other named positions
    2. Have high CONNECTIVITY to other named positions
    3. Lie at INTERSECTIONS of multiple transformation types
    4. Have SYMMETRIC counterparts that are also named
    
    Connection to Zipf's Law:
    - High-frequency words = positions reached by many paths
    - Low-frequency words = positions reached by few paths
    - Unnamed positions = positions with no direct paths
    """)
    
    analyzer = PositionWorthinessAnalyzer()
    
    # Compare named vs unnamed
    print("\n" + "=" * 70)
    print("NAMED vs UNNAMED POSITIONS: AVERAGE METRICS")
    print("=" * 70)
    
    comparison = analyzer.compare_named_vs_unnamed()
    
    print(f"\nNamed positions: {comparison['named_count']}")
    print(f"Unnamed positions: {comparison['unnamed_count']}")
    
    print(f"\n{'Metric':<20} {'Named (avg)':<15} {'Unnamed (avg)':<15} {'Ratio':<10}")
    print("-" * 60)
    
    for metric in ['reachability', 'connectivity', 'symmetry', 'intersections', 'worthiness']:
        named_val = comparison['named'][metric]
        unnamed_val = comparison['unnamed'][metric]
        ratio = named_val / unnamed_val if unnamed_val > 0 else float('inf')
        print(f"{metric:<20} {named_val:<15.2f} {unnamed_val:<15.2f} {ratio:<10.2f}x")
    
    # Show top named positions by worthiness
    print("\n" + "=" * 70)
    print("TOP NAMED POSITIONS BY WORTHINESS")
    print("=" * 70)
    
    ranking = analyzer.get_worthiness_ranking()
    named_ranking = [(a, s) for a, s in ranking if a.is_named]
    
    print(f"\n{'Position':<30} {'Names':<20} {'Score':<10} {'Reach':<8} {'Conn':<8} {'Sym':<6} {'Int':<6}")
    print("-" * 90)
    
    for analysis, score in named_ranking[:15]:
        q = analysis.quaternion
        pos_str = f"({q.x:.1f}, {q.y:.1f}, {q.z:.1f}, {q.w:.1f})"
        names_str = ", ".join(analysis.names[:2])
        if len(analysis.names) > 2:
            names_str += f" +{len(analysis.names)-2}"
        print(f"{pos_str:<30} {names_str:<20} {score:<10.1f} {analysis.reachability:<8} {analysis.connectivity:<8} {analysis.symmetry_score:<6.0f} {analysis.intersection_count:<6}")
    
    # Show unnamed positions with highest worthiness (should be named?)
    print("\n" + "=" * 70)
    print("UNNAMED POSITIONS WITH HIGHEST WORTHINESS (SHOULD BE NAMED?)")
    print("=" * 70)
    
    unnamed_ranking = [(a, s) for a, s in ranking if not a.is_named]
    
    print(f"\n{'Position':<30} {'Score':<10} {'Reach':<8} {'Conn':<8} {'Sym':<6} {'Int':<6} {'Description'}")
    print("-" * 100)
    
    for analysis, score in unnamed_ranking[:10]:
        q = analysis.quaternion
        # Skip extreme positions
        if abs(q.x) > 2 or abs(q.y) > 2 or abs(q.z) > 2 or abs(q.w) > 2:
            continue
        
        pos_str = f"({q.x:.1f}, {q.y:.1f}, {q.z:.1f}, {q.w:.1f})"
        
        # Build description
        desc = []
        if q.x > 0.5: desc.append("male")
        elif q.x < -0.5: desc.append("female")
        if q.y > 0.5: desc.append("adult")
        elif q.y < -0.5: desc.append("young")
        if q.z > 0.5: desc.append("high-agency")
        elif q.z < -0.5: desc.append("low-agency")
        if q.w > 0.5: desc.append("human")
        elif q.w < -0.5: desc.append("abstract")
        
        desc_str = ", ".join(desc) if desc else "neutral"
        
        print(f"{pos_str:<30} {score:<10.1f} {analysis.reachability:<8} {analysis.connectivity:<8} {analysis.symmetry_score:<6.0f} {analysis.intersection_count:<6} {desc_str}")
    
    # Find the naming threshold
    print("\n" + "=" * 70)
    print("THE NAMING THRESHOLD")
    print("=" * 70)
    
    threshold = analyzer.find_naming_threshold()
    print(f"\nEstimated worthiness threshold for naming: {threshold:.2f}")
    print("""
    Positions above this threshold tend to have single-word names.
    Positions below this threshold tend to be lexical gaps.
    
    This threshold represents the "Zipf cutoff" - the point where
    a concept is frequent/important enough to warrant a dedicated word.
    """)
    
    # The key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT: THE GEOMETRY OF NAMING")
    print("=" * 70)
    print("""
    What makes a position "worthy" of a name?
    
    1. REACHABILITY: Can you get there from many other concepts?
       - "woman" is reachable from "man", "girl", "queen", "mother"...
       - A concept with many paths to it gets named
    
    2. CONNECTIVITY: Are your neighbors also named?
       - Named concepts cluster together
       - Isolated positions stay unnamed
    
    3. SYMMETRY: Does your mirror exist?
       - If "king" exists, "queen" should too
       - Asymmetric positions are less stable
    
    4. INTERSECTION: Do multiple transformations meet here?
       - "woman" is at the intersection of gender, age, agency axes
       - Intersections are natural landmarks
    
    THE ZIPF CONNECTION:
    - Word frequency follows power law
    - Position worthiness follows similar distribution
    - The most "worthy" positions get the shortest words
    - Less worthy positions get compounds or stay unnamed
    
    This is why:
    - "king" (high worthiness) = short word
    - "boy king" (lower worthiness) = compound
    - (male, young, high-agency) = no single word
    """)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo()
