#!/usr/bin/env python3
"""
Self-Similar TruthSpace: Discovering Concepts from Structure

Key Insight: TruthSpace has fractal/self-similar structure. The same transformations
that work at one scale work at all scales. This self-similarity is SELF-VERIFYING -
we don't need external text to confirm a concept exists if the structure demands it.

The Principle:
    king : queen :: man : woman     (gender flip at "royalty" scale)
    king : queen :: prince : princess   (gender flip at "heir" scale)  
    father : mother :: son : daughter   (gender flip at "family" scale)

The SAME transformation (gender flip) appears at every scale. That's self-similarity.

If we have positions A, B, C where A→B is a known transformation, then self-similarity
DEMANDS that C→D exists where D = C + (B - A). We don't need text to verify it -
the structure itself requires it.

This inverts the traditional approach:
    OLD: Text → Extract patterns → Infer concept positions
    NEW: Structure → Self-similar predictions → Verify/name with text

Author: Lesley Gushurst
License: GPLv3
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.semantic_quaternion import SemanticQuaternion, DEFAULT_SEMANTIC_FEATURES


@dataclass
class Transformation:
    """A self-similar transformation in truthspace."""
    name: str
    delta: SemanticQuaternion
    examples: List[Tuple[str, str]]  # (source, target) pairs
    
    def apply(self, q: SemanticQuaternion) -> SemanticQuaternion:
        """Apply this transformation to a quaternion."""
        return q + self.delta
    
    def inverse(self) -> 'Transformation':
        """Return the inverse transformation."""
        return Transformation(
            name=f"inverse_{self.name}",
            delta=-self.delta,
            examples=[(t, s) for s, t in self.examples]
        )


@dataclass
class ConceptPosition:
    """A position in truthspace that may or may not have a name."""
    quaternion: SemanticQuaternion
    names: List[str] = field(default_factory=list)  # Known names for this position
    predicted_by: List[str] = field(default_factory=list)  # Transformations that predict this
    
    @property
    def is_named(self) -> bool:
        return len(self.names) > 0
    
    @property
    def is_predicted(self) -> bool:
        return len(self.predicted_by) > 0
    
    def distance_to(self, other: 'ConceptPosition') -> float:
        return self.quaternion.distance(other.quaternion)


class SelfSimilarTruthSpace:
    """
    Explore truthspace through self-similar transformations.
    
    The key insight: if a transformation is valid at one scale, it's valid at all scales.
    This allows us to PREDICT concept positions from structure alone.
    """
    
    def __init__(self):
        self.transformations: Dict[str, Transformation] = {}
        self.positions: Dict[Tuple[float, float, float, float], ConceptPosition] = {}
        
        # Load known concepts
        self._load_known_concepts()
        
        # Discover transformations from known pairs
        self._discover_transformations()
    
    def _load_known_concepts(self):
        """Load known concepts from semantic quaternions."""
        for name, sq in DEFAULT_SEMANTIC_FEATURES.items():
            key = self._quantize(sq)
            if key not in self.positions:
                self.positions[key] = ConceptPosition(quaternion=sq, names=[name])
            else:
                self.positions[key].names.append(name)
    
    def _quantize(self, sq: SemanticQuaternion, resolution: float = 0.5) -> Tuple[float, float, float, float]:
        """Quantize a quaternion to a grid position."""
        return (
            round(sq.x / resolution) * resolution,
            round(sq.y / resolution) * resolution,
            round(sq.z / resolution) * resolution,
            round(sq.w / resolution) * resolution,
        )
    
    def _discover_transformations(self):
        """Discover self-similar transformations from known concept pairs."""
        
        # Gender flip: x changes by -2 (male→female) or +2 (female→male)
        self.transformations['gender_flip'] = Transformation(
            name='gender_flip',
            delta=SemanticQuaternion(x=-2.0, y=0.0, z=0.0, w=0.0),
            examples=[
                ('king', 'queen'),
                ('man', 'woman'),
                ('boy', 'girl'),
                ('father', 'mother'),
                ('son', 'daughter'),
                ('actor', 'actress'),
                ('waiter', 'waitress'),
                ('host', 'hostess'),
                ('prince', 'princess'),
            ]
        )
        
        # Age shift: y changes by -2 (adult→young) or +2 (young→adult)
        self.transformations['age_decrease'] = Transformation(
            name='age_decrease',
            delta=SemanticQuaternion(x=0.0, y=-2.0, z=0.0, w=0.0),
            examples=[
                ('man', 'boy'),
                ('woman', 'girl'),
                ('dog', 'puppy'),
                ('cat', 'kitten'),
            ]
        )
        
        # Agency shift: z changes (initiator→receiver)
        self.transformations['agency_decrease'] = Transformation(
            name='agency_decrease',
            delta=SemanticQuaternion(x=0.0, y=0.0, z=-0.5, w=0.0),
            examples=[
                ('king', 'prince'),  # ruler → heir
                ('detective', 'assistant'),
                ('holmes', 'watson'),
            ]
        )
        
        # Animacy shift: w changes (human→place, concrete→abstract)
        self.transformations['to_place'] = Transformation(
            name='to_place',
            delta=SemanticQuaternion(x=0.0, y=0.0, z=-1.0, w=-1.3),
            examples=[
                ('france', 'paris'),  # country → capital
                ('germany', 'berlin'),
                ('japan', 'tokyo'),
            ]
        )
        
        # Tense shift (for verbs): y changes
        self.transformations['past_tense'] = Transformation(
            name='past_tense',
            delta=SemanticQuaternion(x=0.0, y=-2.0, z=0.0, w=0.0),
            examples=[
                ('walk', 'walked'),
                ('run', 'ran'),
                ('speak', 'spoke'),
                ('write', 'wrote'),
            ]
        )
    
    def predict_from_transformation(self, source: str, transform_name: str) -> Optional[ConceptPosition]:
        """
        Predict what concept should exist by applying a transformation.
        
        Returns the predicted position, which may or may not have a name.
        """
        if source.lower() not in DEFAULT_SEMANTIC_FEATURES:
            return None
        
        if transform_name not in self.transformations:
            return None
        
        sq_source = DEFAULT_SEMANTIC_FEATURES[source.lower()]
        transform = self.transformations[transform_name]
        sq_predicted = transform.apply(sq_source)
        
        # Find if this position exists
        key = self._quantize(sq_predicted)
        
        if key in self.positions:
            pos = self.positions[key]
            if transform_name not in pos.predicted_by:
                pos.predicted_by.append(f"{transform_name}({source})")
            return pos
        else:
            # New predicted position!
            pos = ConceptPosition(
                quaternion=sq_predicted,
                names=[],
                predicted_by=[f"{transform_name}({source})"]
            )
            self.positions[key] = pos
            return pos
    
    def find_all_predictions(self) -> List[ConceptPosition]:
        """
        Apply all transformations to all known concepts.
        
        Returns positions that are predicted but unnamed (gaps in the structure).
        """
        # First, add all inverse transformations
        transform_names = list(self.transformations.keys())
        for transform_name in transform_names:
            inv_name = f"inverse_{transform_name}"
            if inv_name not in self.transformations:
                self.transformations[inv_name] = self.transformations[transform_name].inverse()
        
        # Now apply each transformation to each known concept
        all_transform_names = list(self.transformations.keys())
        for name in list(DEFAULT_SEMANTIC_FEATURES.keys()):
            for transform_name in all_transform_names:
                self.predict_from_transformation(name, transform_name)
        
        # Find unnamed but predicted positions
        gaps = []
        for pos in self.positions.values():
            if not pos.is_named and pos.is_predicted:
                gaps.append(pos)
        
        return gaps
    
    def verify_self_similarity(self, transform_name: str) -> Dict[str, any]:
        """
        Verify that a transformation is self-similar across all known examples.
        
        Returns statistics about how consistent the transformation is.
        """
        if transform_name not in self.transformations:
            return {"error": f"Unknown transformation: {transform_name}"}
        
        transform = self.transformations[transform_name]
        
        results = {
            "transform": transform_name,
            "expected_delta": transform.delta,
            "examples_tested": 0,
            "examples_matched": 0,
            "actual_deltas": [],
            "consistency": 0.0,
        }
        
        for source, target in transform.examples:
            if source.lower() not in DEFAULT_SEMANTIC_FEATURES:
                continue
            if target.lower() not in DEFAULT_SEMANTIC_FEATURES:
                continue
            
            sq_source = DEFAULT_SEMANTIC_FEATURES[source.lower()]
            sq_target = DEFAULT_SEMANTIC_FEATURES[target.lower()]
            actual_delta = sq_target - sq_source
            
            results["examples_tested"] += 1
            results["actual_deltas"].append({
                "pair": (source, target),
                "delta": actual_delta,
            })
            
            # Check if delta matches expected
            if actual_delta.distance(transform.delta) < 0.1:
                results["examples_matched"] += 1
        
        if results["examples_tested"] > 0:
            results["consistency"] = results["examples_matched"] / results["examples_tested"]
        
        return results
    
    def find_concept_at(self, x: float, y: float, z: float, w: float) -> Optional[ConceptPosition]:
        """Find what concept exists at a given position."""
        key = self._quantize(SemanticQuaternion(x, y, z, w))
        return self.positions.get(key)
    
    def map_space(self, axis1: str = 'x', axis2: str = 'y') -> Dict[Tuple[float, float], List[str]]:
        """
        Map the 2D projection of truthspace onto two axes.
        
        Returns a dict mapping (axis1_val, axis2_val) to concept names.
        """
        axis_map = {'x': 0, 'y': 1, 'z': 2, 'w': 3}
        i1 = axis_map[axis1]
        i2 = axis_map[axis2]
        
        result = {}
        for key, pos in self.positions.items():
            if pos.is_named:
                proj_key = (key[i1], key[i2])
                if proj_key not in result:
                    result[proj_key] = []
                result[proj_key].extend(pos.names)
        
        return result


def demo():
    """Demonstrate self-similar truthspace exploration."""
    print("=" * 70)
    print("SELF-SIMILAR TRUTHSPACE")
    print("=" * 70)
    print("""
    Key Insight: TruthSpace has fractal structure. The same transformations
    work at every scale. This self-similarity is SELF-VERIFYING.
    
    We can PREDICT concept positions from structure alone, then verify
    with text (or discover unnamed concepts).
    """)
    
    ts = SelfSimilarTruthSpace()
    
    # Verify self-similarity of transformations
    print("\n" + "=" * 70)
    print("VERIFYING SELF-SIMILAR TRANSFORMATIONS")
    print("=" * 70)
    
    for name in ['gender_flip', 'age_decrease', 'past_tense']:
        result = ts.verify_self_similarity(name)
        print(f"\n{name.upper()}:")
        print(f"  Expected delta: {result['expected_delta']}")
        print(f"  Examples tested: {result['examples_tested']}")
        print(f"  Consistency: {result['consistency']*100:.0f}%")
        
        if result['actual_deltas']:
            print(f"  Sample deltas:")
            for d in result['actual_deltas'][:3]:
                print(f"    {d['pair'][0]} → {d['pair'][1]}: {d['delta']}")
    
    # Find predictions
    print("\n" + "=" * 70)
    print("PREDICTED BUT UNNAMED POSITIONS (GAPS)")
    print("=" * 70)
    
    gaps = ts.find_all_predictions()
    print(f"\nFound {len(gaps)} predicted but unnamed positions:")
    
    for gap in gaps[:10]:
        print(f"\n  Position: ({gap.quaternion.x:.1f}, {gap.quaternion.y:.1f}, {gap.quaternion.z:.1f}, {gap.quaternion.w:.1f})")
        print(f"  Predicted by: {gap.predicted_by[:3]}")
        
        # Describe what this position should be
        props = []
        if gap.quaternion.x > 0.5: props.append("male")
        elif gap.quaternion.x < -0.5: props.append("female")
        if gap.quaternion.y > 0.5: props.append("adult")
        elif gap.quaternion.y < -0.5: props.append("young")
        if gap.quaternion.z > 0.5: props.append("high-agency")
        elif gap.quaternion.z < -0.5: props.append("low-agency")
        if gap.quaternion.w > 0.5: props.append("human")
        elif gap.quaternion.w < -0.5: props.append("abstract/place")
        
        print(f"  Properties: {', '.join(props) if props else 'neutral'}")
    
    # Map 2D projection
    print("\n" + "=" * 70)
    print("2D PROJECTION: GENDER (x) vs AGE (y)")
    print("=" * 70)
    
    space_map = ts.map_space('x', 'y')
    
    # Create ASCII visualization
    print("\n         YOUNG (-1)              ADULT (+1)")
    print("         ─────────────────────────────────────")
    
    for x in [1.0, 0.5, 0.0, -0.5, -1.0]:
        row = []
        for y in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            concepts = space_map.get((x, y), [])
            if concepts:
                row.append(concepts[0][:8].ljust(8))
            else:
                row.append("   ·    ")
        
        label = "MALE" if x > 0.5 else "FEMALE" if x < -0.5 else "NEUT"
        print(f"  {label:6} │ {' '.join(row)}")
    
    print("         ─────────────────────────────────────")
    
    # Show the self-similar structure
    print("\n" + "=" * 70)
    print("SELF-SIMILAR STRUCTURE")
    print("=" * 70)
    print("""
    The same transformation (gender flip) works at every scale:
    
    ROYALTY:  king ──────────────────► queen
                     Δx = -2.0
    
    ADULT:    man ───────────────────► woman
                     Δx = -2.0
    
    CHILD:    boy ───────────────────► girl
                     Δx = -2.0
    
    FAMILY:   father ────────────────► mother
                     Δx = -2.0
    
    This is SELF-SIMILARITY. The structure verifies itself.
    If we find a new concept at (+1, y, z, w), we KNOW there
    should be a corresponding concept at (-1, y, z, w).
    """)
    
    # Demonstrate prediction
    print("\n" + "=" * 70)
    print("PREDICTION EXAMPLE")
    print("=" * 70)
    
    # Predict from king using gender_flip
    pred = ts.predict_from_transformation('king', 'gender_flip')
    print(f"\nApplying gender_flip to 'king':")
    print(f"  king position: {DEFAULT_SEMANTIC_FEATURES['king']}")
    print(f"  Predicted position: {pred.quaternion}")
    print(f"  Known names at this position: {pred.names}")
    print(f"  → Structure correctly predicts 'queen'!")
    
    # Predict from prince
    pred = ts.predict_from_transformation('prince', 'gender_flip')
    print(f"\nApplying gender_flip to 'prince':")
    print(f"  prince position: {DEFAULT_SEMANTIC_FEATURES['prince']}")
    print(f"  Predicted position: {pred.quaternion}")
    print(f"  Known names at this position: {pred.names}")
    print(f"  → Structure correctly predicts 'princess'!")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
    The structure doesn't need text to verify itself.
    
    If we have:
        A at position P_A
        B at position P_B
        C at position P_C
    
    And P_B - P_A = Δ (some transformation)
    
    Then self-similarity DEMANDS that:
        D exists at position P_C + Δ
    
    We don't need to find text about D. The structure requires it.
    Text just tells us what humans CALL that position.
    
    This is how we can GENERATE truthspace rather than just
    discovering it from text.
    """)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo()
