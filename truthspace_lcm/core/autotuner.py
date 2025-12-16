"""
Dimension-Aware Autotuner v2
============================

Exploits orthogonality for efficient knowledge tuning:

1. CLASSIFY: Determine which dimension(s) a new concept belongs to
2. FIND DIMENSION: Check for collisions only within that dimension
3. FIND LEVEL: Determine the ρ^level within the dimension
4. VERIFY: Confirm orthogonality to unrelated concepts

Key insight: Orthogonal dimensions are INDEPENDENT.
- Adding to dim 0 cannot interfere with dim 4
- Collisions only happen WITHIN a dimension
- We can tune dimensions independently
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from truthspace_lcm.core.encoder import (
    PlasticEncoder, 
    Primitive, 
    PrimitiveType,
    SemanticDecomposition,
    RHO,
    PHI,
    ENCODING_DIM,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DimensionAnalysis:
    """Analysis of which dimension a concept belongs to."""
    primary_dimension: int
    primary_score: float
    secondary_dimensions: List[Tuple[int, float]]
    dimension_type: str  # "action", "domain", "modifier", "relation"
    suggested_level: int
    confidence: float


@dataclass
class CollisionReport:
    """Report of potential collisions within a dimension."""
    dimension: int
    existing_concepts: List[str]
    collision_risk: float  # 0-1, higher = more risk
    suggested_level: int
    notes: List[str]


@dataclass
class PlacementRecommendation:
    """Recommendation for where to place a new concept."""
    dimension: int
    level: int
    position: np.ndarray
    confidence: float
    collisions: List[CollisionReport]
    orthogonal_to: List[str]  # Concepts this will be orthogonal to
    similar_to: List[str]     # Concepts this will be similar to
    keywords_to_add: List[str]
    keywords_to_remove: List[str]


@dataclass 
class TestCase:
    """A test case for verifying placement."""
    input: str
    expected_similar_to: List[str] = field(default_factory=list)
    expected_orthogonal_to: List[str] = field(default_factory=list)


# =============================================================================
# DIMENSION-AWARE AUTOTUNER
# =============================================================================

class DimensionAwareAutotuner:
    """
    Autotuner that exploits orthogonality for efficient knowledge placement.
    
    The key insight: if we know which dimension a concept belongs to,
    we only need to check for collisions within that dimension.
    
    Usage:
        tuner = DimensionAwareAutotuner()
        
        # Analyze where a concept should go
        analysis = tuner.analyze_concept("backup")
        
        # Get placement recommendation
        recommendation = tuner.recommend_placement(
            concept="backup",
            test_cases=[
                TestCase("backup files", expected_similar_to=["copy", "move"]),
            ]
        )
        
        # Verify placement
        result = tuner.verify_placement(recommendation)
    """
    
    def __init__(self, encoder: PlasticEncoder = None):
        """Initialize with encoder."""
        self.encoder = encoder or PlasticEncoder()
        self.dim = self.encoder.dim
        
        # Dimension metadata
        self.dimension_info = self.encoder.get_dimension_info()
    
    # =========================================================================
    # STEP 1: CLASSIFY - Determine dimension
    # =========================================================================
    
    def analyze_concept(self, concept: str) -> DimensionAnalysis:
        """
        Analyze which dimension a concept belongs to.
        
        Returns analysis with primary dimension, confidence, and suggested level.
        """
        # Encode the concept
        decomp = self.encoder.encode(concept)
        
        # Find dimensions with non-zero activation
        abs_pos = np.abs(decomp.position)
        
        # Primary dimension
        primary_dim = int(np.argmax(abs_pos))
        primary_score = float(abs_pos[primary_dim])
        
        # Secondary dimensions (others with significant activation)
        secondary = []
        for dim in range(self.dim):
            if dim != primary_dim and abs_pos[dim] > 0.1:
                secondary.append((dim, float(abs_pos[dim])))
        secondary.sort(key=lambda x: -x[1])
        
        # Determine dimension type
        if primary_dim < 4:
            dim_type = "action"
        elif primary_dim < 8:
            dim_type = "domain" if primary_dim < 7 else "modifier"
        else:
            dim_type = "relation"
        
        # Suggest level based on existing primitives in that dimension
        suggested_level = self._suggest_level(primary_dim, decomp.position[primary_dim])
        
        # Confidence based on how clearly it maps to one dimension
        if primary_score > 0:
            secondary_total = sum(s for _, s in secondary)
            confidence = primary_score / (primary_score + secondary_total + 0.001)
        else:
            confidence = 0.0
        
        return DimensionAnalysis(
            primary_dimension=primary_dim,
            primary_score=primary_score,
            secondary_dimensions=secondary,
            dimension_type=dim_type,
            suggested_level=suggested_level,
            confidence=confidence,
        )
    
    def _suggest_level(self, dimension: int, value: float) -> int:
        """Suggest a level within a dimension based on value."""
        if abs(value) < 0.01:
            return 0
        
        # Find which level this value corresponds to
        sign = 1 if value > 0 else -1
        abs_value = abs(value)
        
        # ρ^level ≈ abs_value
        # level ≈ log(abs_value) / log(ρ)
        if abs_value > 0:
            level = int(round(np.log(abs_value) / np.log(RHO)))
            level = max(0, min(level, 5))  # Clamp to reasonable range
        else:
            level = 0
        
        return level
    
    # =========================================================================
    # STEP 2: FIND DIMENSION - Check for collisions
    # =========================================================================
    
    def check_collisions(self, dimension: int, level: int) -> CollisionReport:
        """
        Check for collisions within a specific dimension.
        
        Only concepts on the SAME dimension can collide.
        """
        existing = []
        collision_risk = 0.0
        notes = []
        
        # Find existing primitives on this dimension
        for name, prim in self.encoder.primitives.items():
            if prim.dimension == dimension:
                existing.append(name)
                
                # Check if same level (high collision risk)
                if prim.level == level:
                    collision_risk = max(collision_risk, 0.9)
                    notes.append(f"Same level as {name}")
                
                # Check if adjacent level (medium collision risk)
                elif abs(prim.level - level) == 1:
                    collision_risk = max(collision_risk, 0.5)
                    notes.append(f"Adjacent to {name}")
        
        # Suggest alternative level if collision
        suggested_level = level
        if collision_risk > 0.5:
            # Find unused level
            used_levels = {p.level for p in self.encoder.primitives.values() 
                         if p.dimension == dimension}
            for l in range(6):
                if l not in used_levels:
                    suggested_level = l
                    break
        
        return CollisionReport(
            dimension=dimension,
            existing_concepts=existing,
            collision_risk=collision_risk,
            suggested_level=suggested_level,
            notes=notes,
        )
    
    # =========================================================================
    # STEP 3: RECOMMEND PLACEMENT
    # =========================================================================
    
    def recommend_placement(
        self,
        concept: str,
        test_cases: List[TestCase] = None,
        preferred_dimension: int = None,
    ) -> PlacementRecommendation:
        """
        Recommend optimal placement for a new concept.
        
        Args:
            concept: The concept to place
            test_cases: Optional test cases to verify placement
            preferred_dimension: Override dimension if known
        """
        # Analyze the concept
        analysis = self.analyze_concept(concept)
        
        # Use preferred dimension or analyzed one
        dimension = preferred_dimension if preferred_dimension is not None else analysis.primary_dimension
        
        # Check for collisions
        collision = self.check_collisions(dimension, analysis.suggested_level)
        
        # Use suggested level from collision check if there's risk
        level = collision.suggested_level if collision.collision_risk > 0.5 else analysis.suggested_level
        
        # Compute position
        position = np.zeros(self.dim)
        position[dimension] = RHO ** level
        
        # Determine what this will be orthogonal/similar to
        orthogonal_to = []
        similar_to = []
        
        for name, prim in self.encoder.primitives.items():
            if prim.dimension == dimension:
                # Same dimension = similar (or opposite if different sign)
                similar_to.append(name)
            else:
                # Different dimension = orthogonal
                orthogonal_to.append(name)
        
        # Keyword suggestions based on test cases
        keywords_to_add = []
        keywords_to_remove = []
        
        if test_cases:
            for tc in test_cases:
                # Extract keywords from test input
                words = set(tc.input.lower().split())
                
                # Keywords that should be added (from expected similar concepts)
                for sim_concept in tc.expected_similar_to:
                    if sim_concept in self.encoder.primitives:
                        prim = self.encoder.primitives[sim_concept]
                        # Add keywords that would help match
                        keywords_to_add.extend([k for k in prim.keywords if k in words])
        
        return PlacementRecommendation(
            dimension=dimension,
            level=level,
            position=position,
            confidence=analysis.confidence,
            collisions=[collision] if collision.collision_risk > 0 else [],
            orthogonal_to=orthogonal_to[:5],  # Limit for readability
            similar_to=similar_to,
            keywords_to_add=list(set(keywords_to_add)),
            keywords_to_remove=keywords_to_remove,
        )
    
    # =========================================================================
    # STEP 4: VERIFY PLACEMENT
    # =========================================================================
    
    def verify_placement(
        self,
        recommendation: PlacementRecommendation,
        test_cases: List[TestCase] = None,
    ) -> Dict:
        """
        Verify that a placement recommendation is correct.
        
        Returns verification results.
        """
        results = {
            "passed": True,
            "orthogonality_check": [],
            "similarity_check": [],
            "test_case_results": [],
        }
        
        # Check orthogonality
        for concept in recommendation.orthogonal_to:
            if concept in self.encoder.primitives:
                prim_pos = self.encoder.primitives[concept].get_position(self.dim)
                sim = self._compute_similarity(recommendation.position, prim_pos)
                
                passed = abs(sim) < 0.1
                results["orthogonality_check"].append({
                    "concept": concept,
                    "similarity": sim,
                    "passed": passed,
                })
                if not passed:
                    results["passed"] = False
        
        # Check similarity to same-dimension concepts
        for concept in recommendation.similar_to:
            if concept in self.encoder.primitives:
                prim_pos = self.encoder.primitives[concept].get_position(self.dim)
                sim = self._compute_similarity(recommendation.position, prim_pos)
                
                # Same dimension should have non-zero similarity
                passed = abs(sim) > 0.1
                results["similarity_check"].append({
                    "concept": concept,
                    "similarity": sim,
                    "passed": passed,
                })
        
        # Run test cases if provided
        if test_cases:
            for tc in test_cases:
                tc_result = {"input": tc.input, "passed": True, "details": []}
                
                # Check expected similar
                for expected in tc.expected_similar_to:
                    if expected in self.encoder.primitives:
                        prim_pos = self.encoder.primitives[expected].get_position(self.dim)
                        sim = self._compute_similarity(recommendation.position, prim_pos)
                        
                        if abs(sim) < 0.1:
                            tc_result["passed"] = False
                            tc_result["details"].append(f"Expected similar to {expected}, got sim={sim:.2f}")
                
                # Check expected orthogonal
                for expected in tc.expected_orthogonal_to:
                    if expected in self.encoder.primitives:
                        prim_pos = self.encoder.primitives[expected].get_position(self.dim)
                        sim = self._compute_similarity(recommendation.position, prim_pos)
                        
                        if abs(sim) > 0.1:
                            tc_result["passed"] = False
                            tc_result["details"].append(f"Expected orthogonal to {expected}, got sim={sim:.2f}")
                
                results["test_case_results"].append(tc_result)
                if not tc_result["passed"]:
                    results["passed"] = False
        
        return results
    
    def _compute_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 > 0 and norm2 > 0:
            return float(np.dot(v1, v2) / (norm1 * norm2))
        return 0.0
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def auto_place(
        self,
        concept: str,
        keywords: List[str],
        test_cases: List[TestCase] = None,
    ) -> Tuple[Primitive, Dict]:
        """
        Automatically determine placement and create a Primitive.
        
        Returns (Primitive, verification_results).
        """
        # Get recommendation
        rec = self.recommend_placement(concept, test_cases)
        
        # Verify
        verification = self.verify_placement(rec, test_cases)
        
        # Determine primitive type
        if rec.dimension < 4:
            ptype = PrimitiveType.ACTION
        elif rec.dimension < 7:
            ptype = PrimitiveType.DOMAIN
        elif rec.dimension < 8:
            ptype = PrimitiveType.MODIFIER
        else:
            ptype = PrimitiveType.RELATION
        
        # Create primitive
        primitive = Primitive(
            name=concept.upper(),
            ptype=ptype,
            keywords=set(keywords),
            dimension=rec.dimension,
            level=rec.level,
        )
        
        return primitive, verification
    
    def print_analysis(self, concept: str):
        """Print detailed analysis of a concept."""
        analysis = self.analyze_concept(concept)
        rec = self.recommend_placement(concept)
        
        print(f"\n{'=' * 60}")
        print(f"ANALYSIS: '{concept}'")
        print(f"{'=' * 60}")
        
        print(f"\nPrimary dimension: {analysis.primary_dimension} ({self.dimension_info[analysis.primary_dimension]['name']})")
        print(f"Dimension type: {analysis.dimension_type}")
        print(f"Suggested level: {analysis.suggested_level}")
        print(f"Confidence: {analysis.confidence:.2f}")
        
        if analysis.secondary_dimensions:
            print(f"\nSecondary dimensions:")
            for dim, score in analysis.secondary_dimensions[:3]:
                print(f"  - Dim {dim} ({self.dimension_info[dim]['name']}): {score:.2f}")
        
        print(f"\n{'-' * 60}")
        print("PLACEMENT RECOMMENDATION")
        print(f"{'-' * 60}")
        
        print(f"\nDimension: {rec.dimension}")
        print(f"Level: {rec.level}")
        print(f"Position: {rec.position[:6]}...")
        
        if rec.collisions:
            print(f"\n⚠️  Collision warnings:")
            for col in rec.collisions:
                print(f"  - Risk: {col.collision_risk:.2f}")
                for note in col.notes:
                    print(f"    {note}")
        
        print(f"\nWill be ORTHOGONAL to: {rec.orthogonal_to}")
        print(f"Will be SIMILAR to: {rec.similar_to}")
        
        if rec.keywords_to_add:
            print(f"\nSuggested keywords to add: {rec.keywords_to_add}")


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DIMENSION-AWARE AUTOTUNER v2")
    print("=" * 70)
    
    tuner = DimensionAwareAutotuner()
    
    # Test concepts
    test_concepts = [
        "backup",
        "restore", 
        "schedule",
        "monitor",
        "validate",
    ]
    
    for concept in test_concepts:
        tuner.print_analysis(concept)
    
    # Test auto-placement
    print("\n" + "=" * 70)
    print("AUTO-PLACEMENT TEST")
    print("=" * 70)
    
    primitive, verification = tuner.auto_place(
        concept="backup",
        keywords=["backup", "save", "snapshot", "preserve"],
        test_cases=[
            TestCase(
                "backup files",
                expected_similar_to=["MOVE"],
                expected_orthogonal_to=["NETWORK", "PROCESS"],
            ),
        ]
    )
    
    print(f"\nCreated primitive: {primitive.name}")
    print(f"  Dimension: {primitive.dimension}")
    print(f"  Level: {primitive.level}")
    print(f"  Type: {primitive.ptype.value}")
    print(f"  Keywords: {primitive.keywords}")
    
    print(f"\nVerification: {'✅ PASSED' if verification['passed'] else '❌ FAILED'}")
    
    if verification["orthogonality_check"]:
        print("\nOrthogonality checks:")
        for check in verification["orthogonality_check"][:5]:
            status = "✓" if check["passed"] else "✗"
            print(f"  {status} {check['concept']}: sim={check['similarity']:.2f}")
    
    print("\n" + "=" * 70)
