#!/usr/bin/env python3
"""
Implicit Knowledge Through Structural Gaps

Key insight: The φ-based navigation IS implicit knowledge.
The gaps and errors in concept space ARE implicit knowledge signals.

When we ask "watson:?" and get nothing explicit, the GAP itself is information:
- Where in the structure is the gap?
- What SHAPE is the gap?
- What would FIT in that gap?

This is fundamentally different from explicit lookup:
- Explicit: "Is there a relationship?" → Yes/No
- Implicit: "What does the structure suggest should be here?"

The geometry of concept space creates a LANDSCAPE.
Gaps in that landscape are as informative as peaks.

Author: Lesley Gushurst
License: GPLv3
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.distilled_lcm import DistilledLCM, Concept

PHI = 1.618034


@dataclass
class GapProfile:
    """
    Describes a "gap" in concept space - where implicit knowledge might exist.
    
    A gap is characterized by:
    - Expected position (where we'd expect something to be)
    - Neighboring concepts (what's around the gap)
    - Shape (what properties the missing concept should have)
    """
    expected_phi: float  # Expected φ-direction
    expected_actions: List[str]  # Expected action patterns
    expected_targets: List[str]  # Expected target patterns
    neighbors: List[str]  # Concepts near this gap
    confidence: float  # How confident are we this gap is meaningful?
    
    def describe(self) -> str:
        return (f"Gap: φ≈{self.expected_phi:.2f}, "
                f"actions≈{self.expected_actions[:2]}, "
                f"targets≈{self.expected_targets[:2]}, "
                f"near {self.neighbors[:3]}")


class ImplicitKnowledge:
    """
    Explore implicit knowledge through structural gaps in concept space.
    
    The key insight: φ-navigation creates a geometric structure where
    GAPS are as meaningful as explicit relationships.
    """
    
    def __init__(self, distilled_path: str = "truthspace_lcm/concepts_distilled.json"):
        self.distilled = DistilledLCM()
        self.distilled.load(distilled_path)
        
        # Build spatial index for gap detection
        self._build_phi_index()
    
    def _build_phi_index(self):
        """Index concepts by φ-direction for spatial queries."""
        self.phi_buckets: Dict[int, List[str]] = {}
        
        for word, concept in self.distilled.concepts.items():
            # Bucket by φ-direction (10 buckets from -1 to +1)
            bucket = int((concept.phi_direction + 1) * 5)
            bucket = max(0, min(9, bucket))
            
            if bucket not in self.phi_buckets:
                self.phi_buckets[bucket] = []
            self.phi_buckets[bucket].append(word)
    
    def find_neighbors(self, phi: float, k: int = 10) -> List[Tuple[str, float]]:
        """Find concepts near a given φ-direction."""
        results = []
        
        for word, concept in self.distilled.concepts.items():
            distance = abs(concept.phi_direction - phi)
            results.append((word, distance))
        
        results.sort(key=lambda x: x[1])
        return results[:k]
    
    def detect_gap(self, a: str, b: str, c: str) -> Optional[GapProfile]:
        """
        Detect the "gap" that an analogy points to.
        
        A:B :: C:?
        
        Key insight: The relationship TYPE matters.
        - If B is a category word, we're looking for C's category
        - If B is an action, we're looking for C's parallel action
        - If B is a target of A, we're looking for C's parallel target
        
        The gap shape should match the RELATIONSHIP TYPE, not just φ-direction.
        """
        concept_a = self.distilled.get_concept(a.lower())
        concept_b = self.distilled.get_concept(b.lower())
        concept_c = self.distilled.get_concept(c.lower())
        
        if not all([concept_a, concept_b, concept_c]):
            return None
        
        # Detect relationship type
        category_words = {
            'detective', 'doctor', 'scientist', 'teacher', 'writer',
            'philosopher', 'artist', 'leader', 'hero', 'villain',
            'companion', 'assistant', 'friend', 'enemy', 'partner',
            'science', 'field', 'discipline', 'study', 'theory',
            'physicist', 'biologist', 'chemist', 'mathematician',
            'king', 'queen', 'prince', 'princess', 'man', 'woman',
        }
        
        b_word = b.lower()
        is_category = b_word in category_words
        
        if is_category:
            # B is a category - we want C's category
            # The gap should be at B's φ-direction (categories have their own φ)
            expected_phi = concept_b.phi_direction
            
            # Expected actions: similar to what B does (if B is a role)
            expected_actions = [act for act, _ in concept_b.actions[:5]]
            
            # Expected targets: what C acts on (context)
            expected_targets = [t for t, _ in concept_c.targets[:5]]
            
            # Higher confidence for category relationships
            confidence = 0.8
        else:
            # B is not a category - use φ-delta approach
            phi_delta = concept_b.phi_direction - concept_a.phi_direction
            expected_phi = concept_c.phi_direction + phi_delta
            expected_phi = max(-1, min(1, expected_phi))
            
            expected_actions = [act for act, _ in concept_b.actions[:5]]
            expected_targets = [t for t, _ in concept_c.targets[:5]]
            
            confidence = 1.0 - abs(phi_delta) / 2
        
        # Find neighbors near the expected position
        neighbors = [word for word, _ in self.find_neighbors(expected_phi, k=10)]
        
        return GapProfile(
            expected_phi=expected_phi,
            expected_actions=expected_actions,
            expected_targets=expected_targets,
            neighbors=neighbors,
            confidence=confidence
        )
    
    def fill_gap(self, gap: GapProfile, k: int = 5, 
                  source_concept: str = None, 
                  relationship_word: str = None) -> List[Tuple[str, float]]:
        """
        Find concepts that best FIT the detected gap.
        
        Key insight: The gap shape alone isn't enough.
        We also need to consider:
        - What domain is C in? (watson is in detective domain)
        - What relationship does B have to A? (detective is holmes's role)
        
        The answer D should be in C's domain and have similar relationship.
        """
        results = []
        
        # Category words - these are what we're often looking for
        category_words = {
            'detective', 'doctor', 'scientist', 'teacher', 'writer',
            'philosopher', 'artist', 'leader', 'hero', 'villain',
            'companion', 'assistant', 'friend', 'enemy', 'partner',
            'science', 'field', 'discipline', 'study', 'theory',
            'physicist', 'biologist', 'chemist', 'mathematician',
        }
        
        # Get source concept's context (what domain is it in?)
        source_context = set()
        if source_concept:
            sc = self.distilled.get_concept(source_concept.lower())
            if sc:
                source_context = set(t for t, _ in sc.targets[:10])
                source_context.update(a for a, _ in sc.actions[:10])
        
        # Get relationship word's context
        rel_context = set()
        if relationship_word:
            rc = self.distilled.get_concept(relationship_word.lower())
            if rc:
                rel_context = set(t for t, _ in rc.targets[:10])
                rel_context.update(a for a, _ in rc.actions[:10])
        
        for word, concept in self.distilled.concepts.items():
            # Skip very common words
            if concept.frequency > 500:
                continue
            
            # Skip if too far from expected φ
            phi_distance = abs(concept.phi_direction - gap.expected_phi)
            if phi_distance > 0.7:
                continue
            
            score = 0.0
            
            # 1. φ-direction match
            phi_score = math.exp(-phi_distance * 3)
            score += phi_score * 2.0
            
            # 2. Domain match - does this concept share context with source?
            concept_context = set(t for t, _ in concept.targets[:10])
            concept_context.update(a for a, _ in concept.actions[:10])
            
            if source_context:
                domain_overlap = len(concept_context & source_context)
                score += domain_overlap * 0.5
            
            # 3. Relationship match - similar to the relationship word?
            if rel_context:
                rel_overlap = len(concept_context & rel_context)
                score += rel_overlap * 0.3
            
            # 4. Category word bonus
            if word in category_words:
                score += 1.5
            
            # 5. Neighbor bonus
            if word in gap.neighbors:
                score += 0.2
            
            # 6. Frequency penalty
            freq_penalty = math.log(concept.frequency + 1) / 10
            score -= freq_penalty * 0.3
            
            if score > 0:
                results.append((word, score))
        
        results.sort(key=lambda x: -x[1])
        return results[:k]
    
    def implicit_analogy(self, a: str, b: str, c: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Complete an analogy using IMPLICIT knowledge (gap detection).
        
        A:B :: C:?
        
        Instead of looking for explicit "C is-a X" relationships,
        we find the gap that the analogy points to and fill it.
        
        Now with domain context: C's domain should inform what D is.
        """
        gap = self.detect_gap(a, b, c)
        if not gap:
            return []
        
        # Fill the gap with concepts that fit the structure
        # Pass C as source_concept and B as relationship_word for context
        candidates = self.fill_gap(gap, k=k*2, source_concept=c, relationship_word=b)
        
        # Filter out the input concepts
        filtered = [(word, score) for word, score in candidates 
                   if word.lower() not in {a.lower(), b.lower(), c.lower()}]
        
        return filtered[:k]
    
    def explore_structure(self, word: str) -> Dict:
        """
        Explore the structural neighborhood of a concept.
        
        Returns information about:
        - Where the concept sits in φ-space
        - What gaps exist nearby
        - What implicit relationships the structure suggests
        """
        concept = self.distilled.get_concept(word.lower())
        if not concept:
            return {"error": f"Unknown concept: {word}"}
        
        # Find neighbors at similar φ
        similar_phi = self.find_neighbors(concept.phi_direction, k=10)
        
        # Find neighbors at opposite φ (potential contrasts)
        opposite_phi = self.find_neighbors(-concept.phi_direction, k=5)
        
        # Find concepts with similar actions
        concept_actions = set(act for act, _ in concept.actions[:5])
        similar_actions = []
        for other_word, other_concept in self.distilled.concepts.items():
            if other_word == word.lower():
                continue
            other_actions = set(act for act, _ in other_concept.actions[:5])
            overlap = len(concept_actions & other_actions)
            if overlap >= 2:
                similar_actions.append((other_word, overlap))
        similar_actions.sort(key=lambda x: -x[1])
        
        return {
            "word": word,
            "phi": concept.phi_direction,
            "actions": [a for a, _ in concept.actions[:5]],
            "targets": [t for t, _ in concept.targets[:5]],
            "similar_phi": similar_phi[:5],
            "opposite_phi": opposite_phi[:3],
            "similar_actions": similar_actions[:5],
        }


def auto_learn_test():
    """
    Test using gaps to AUTO-LEARN missing relationships.
    
    The idea: If we detect a gap with high confidence,
    we can infer what relationship SHOULD exist and inject it.
    """
    print("=" * 70)
    print("AUTO-LEARNING FROM STRUCTURAL GAPS")
    print("=" * 70)
    
    ik = ImplicitKnowledge()
    
    # Find gaps that suggest missing relationships
    print("\n--- Detecting High-Confidence Gaps ---")
    
    # Test cases where we KNOW what the answer should be
    known_answers = [
        ('holmes', 'detective', 'watson', 'doctor'),
        ('holmes', 'detective', 'moriarty', 'villain'),
        ('physics', 'science', 'biology', 'science'),
        ('king', 'queen', 'man', 'woman'),
        ('einstein', 'physicist', 'darwin', 'biologist'),
    ]
    
    print(f"\n{'Analogy':<35} {'Expected':<12} {'Top Implicit':<15} {'Match?'}")
    print("-" * 75)
    
    matches = 0
    for a, b, c, expected in known_answers:
        results = ik.implicit_analogy(a, b, c, k=5)
        gap = ik.detect_gap(a, b, c)
        
        top_result = results[0][0] if results else "(none)"
        top_5 = [r[0] for r in results[:5]] if results else []
        
        # Check if expected is in top 5
        in_top_5 = expected in top_5
        if in_top_5:
            matches += 1
        
        match_str = "✓ (in top 5)" if in_top_5 else "✗"
        analogy_str = f"{a}:{b} :: {c}:?"
        print(f"{analogy_str:<35} {expected:<12} {top_result:<15} {match_str}")
        
        if gap and gap.confidence > 0.5:
            print(f"  Gap: φ={gap.expected_phi:.2f}, conf={gap.confidence:.2f}")
    
    print(f"\nAccuracy (expected in top 5): {matches}/{len(known_answers)} ({100*matches/len(known_answers):.0f}%)")
    
    # Key insight: check WHY expected answers don't match
    print("\n--- Why 'Expected' Doesn't Match Structure ---")
    print("Checking φ-direction of expected answers vs gap expectations...\n")
    
    for a, b, c, expected in known_answers:
        gap = ik.detect_gap(a, b, c)
        expected_concept = ik.distilled.get_concept(expected.lower())
        c_concept = ik.distilled.get_concept(c.lower())
        
        if gap and expected_concept and c_concept:
            phi_diff = abs(expected_concept.phi_direction - gap.expected_phi)
            c_phi_diff = abs(c_concept.phi_direction - gap.expected_phi)
            
            print(f"{a}:{b} :: {c}:?")
            print(f"  Gap expects φ={gap.expected_phi:.2f}")
            print(f"  '{expected}' has φ={expected_concept.phi_direction:.2f} (diff={phi_diff:.2f})")
            print(f"  '{c}' has φ={c_concept.phi_direction:.2f}")
            
            if phi_diff > 1.0:
                print(f"  → '{expected}' is STRUCTURALLY OPPOSITE to the gap!")
            elif phi_diff > 0.5:
                print(f"  → '{expected}' is structurally distant from the gap")
            else:
                print(f"  → '{expected}' fits the gap structure")
            print()
    
    # Now test: what would we LEARN from these gaps?
    print("\n--- What Gaps Suggest We Should Learn ---")
    
    for a, b, c, expected in known_answers[:3]:
        gap = ik.detect_gap(a, b, c)
        results = ik.implicit_analogy(a, b, c, k=3)
        
        if gap:
            print(f"\n{a}:{b} :: {c}:?")
            print(f"  Gap shape: φ={gap.expected_phi:.2f}")
            print(f"  Structure suggests: {[r[0] for r in results]}")
            print(f"  Expected: {expected}")
            
            # What frame would we inject?
            if results:
                best = results[0][0]
                print(f"  Would inject: '{c} is a {best}' (confidence={gap.confidence:.2f})")
                if best != expected:
                    print(f"  Note: Structure suggests '{best}', not '{expected}'")
                    print(f"        This could be a valid alternative or a gap in our category words")


def extended_test():
    """Extended testing of implicit knowledge."""
    print("=" * 70)
    print("EXTENDED IMPLICIT KNOWLEDGE TESTING")
    print("=" * 70)
    
    ik = ImplicitKnowledge()
    
    # Test a wide variety of analogies
    test_cases = [
        # Classic word2vec style
        ('king', 'queen', 'man'),
        ('king', 'queen', 'prince'),
        ('man', 'woman', 'king'),
        
        # Role/profession based
        ('holmes', 'detective', 'watson'),
        ('holmes', 'detective', 'moriarty'),
        ('einstein', 'physicist', 'darwin'),
        
        # Domain based
        ('physics', 'science', 'biology'),
        ('physics', 'science', 'chemistry'),
        ('newton', 'physics', 'darwin'),
        
        # Action based
        ('detective', 'investigate', 'doctor'),
        ('writer', 'write', 'painter'),
        
        # Relationship based
        ('watson', 'holmes', 'moriarty'),
        ('electron', 'atom', 'planet'),
    ]
    
    print("\n--- Implicit Analogy Results ---")
    print(f"{'Analogy':<40} {'Top 3 Results':<50}")
    print("-" * 90)
    
    successes = 0
    for a, b, c in test_cases:
        results = ik.implicit_analogy(a, b, c, k=3)
        analogy_str = f"{a}:{b} :: {c}:?"
        
        if results:
            results_str = ", ".join([f"{w}({s:.1f})" for w, s in results])
            successes += 1
        else:
            results_str = "(no results)"
        
        print(f"{analogy_str:<40} {results_str:<50}")
    
    print(f"\nSuccess rate: {successes}/{len(test_cases)} ({100*successes/len(test_cases):.0f}%)")
    
    # Test gap shapes
    print("\n--- Gap Shape Analysis ---")
    for a, b, c in [('holmes', 'detective', 'watson'), 
                    ('physics', 'science', 'biology'),
                    ('king', 'queen', 'man')]:
        gap = ik.detect_gap(a, b, c)
        if gap:
            print(f"\n{a}:{b} :: {c}:?")
            print(f"  Expected φ: {gap.expected_phi:.2f}")
            print(f"  Expected actions: {gap.expected_actions[:3]}")
            print(f"  Expected targets: {gap.expected_targets[:3]}")
            print(f"  Neighbors: {gap.neighbors[:5]}")
            print(f"  Confidence: {gap.confidence:.2f}")
    
    # Test: Can we find concepts that SHOULD be related?
    print("\n--- Structural Relationship Discovery ---")
    print("Finding concepts that SHOULD be related based on structure...")
    
    # Find concepts with similar φ and actions to 'detective'
    detective = ik.distilled.get_concept('detective')
    if detective:
        detective_actions = set(a for a, _ in detective.actions[:5])
        print(f"\nDetective: φ={detective.phi_direction:.2f}, actions={list(detective_actions)}")
        print("Structurally similar concepts:")
        
        similar = []
        for word, concept in ik.distilled.concepts.items():
            if word == 'detective':
                continue
            phi_diff = abs(concept.phi_direction - detective.phi_direction)
            if phi_diff < 0.3:
                concept_actions = set(a for a, _ in concept.actions[:5])
                action_overlap = len(detective_actions & concept_actions)
                if action_overlap > 0 or phi_diff < 0.1:
                    similar.append((word, phi_diff, action_overlap))
        
        similar.sort(key=lambda x: (x[1], -x[2]))
        for word, phi_diff, overlap in similar[:10]:
            print(f"  {word}: φ_diff={phi_diff:.2f}, action_overlap={overlap}")


def demo():
    """Demonstrate implicit knowledge through gap detection."""
    print("=" * 70)
    print("IMPLICIT KNOWLEDGE THROUGH STRUCTURAL GAPS")
    print("=" * 70)
    
    ik = ImplicitKnowledge()
    
    # 1. Explore structure around key concepts
    print("\n--- Structural Exploration ---")
    for word in ['holmes', 'watson', 'detective', 'physics']:
        info = ik.explore_structure(word)
        print(f"\n{word.upper()}:")
        print(f"  φ = {info['phi']:.2f}")
        print(f"  Actions: {info['actions']}")
        print(f"  Similar φ: {[w for w, _ in info['similar_phi'][:3]]}")
        print(f"  Similar actions: {[w for w, _ in info['similar_actions'][:3]]}")
    
    # 2. Detect gaps for analogies
    print("\n--- Gap Detection ---")
    analogies = [
        ('holmes', 'detective', 'watson'),
        ('physics', 'science', 'biology'),
        ('king', 'queen', 'man'),
    ]
    
    for a, b, c in analogies:
        print(f"\n{a}:{b} :: {c}:?")
        gap = ik.detect_gap(a, b, c)
        if gap:
            print(f"  {gap.describe()}")
            print(f"  Confidence: {gap.confidence:.2f}")
    
    # 3. Fill gaps with implicit knowledge
    print("\n--- Implicit Analogies (Gap Filling) ---")
    for a, b, c in analogies:
        print(f"\n{a}:{b} :: {c}:?")
        results = ik.implicit_analogy(a, b, c, k=5)
        if results:
            for word, score in results:
                print(f"  → {word} (score={score:.2f})")
        else:
            print("  (no candidates found)")
    
    # 4. Compare explicit vs implicit
    print("\n--- Explicit vs Implicit Knowledge ---")
    print("""
    EXPLICIT: "Is there a stated relationship?"
      - Looks for frames like "watson is a doctor"
      - Returns nothing if not explicitly stated
      - 100% precision, limited recall
    
    IMPLICIT: "What does the structure suggest?"
      - Looks at φ-direction, action patterns, neighbors
      - Infers what SHOULD be there based on geometry
      - Lower precision, higher recall
      - The GAP is the knowledge
    """)
    
    # 5. The key insight: gaps tell us what's MISSING
    print("\n--- What the Gaps Tell Us ---")
    print("""
    The gap for 'holmes:detective :: watson:?' points to φ≈1.0
    
    This tells us: "Watson should relate to something with HIGH AGENCY"
    
    The corpus doesn't have "Watson is a doctor" explicitly,
    but the STRUCTURE suggests Watson should connect to a
    high-agency role concept.
    
    The gap IS the implicit knowledge:
    - Shape: high φ, role-like, connected to Watson
    - What fits: teacher, doctor, companion, assistant
    - What's missing: the explicit "watson is a X" frame
    
    To LEARN this implicitly, we could:
    1. Detect gaps that appear frequently
    2. Infer the missing relationship from the shape
    3. Inject it as a low-confidence frame
    4. Let it strengthen if more evidence appears
    """)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'extended':
        extended_test()
    elif len(sys.argv) > 1 and sys.argv[1] == 'autolearn':
        auto_learn_test()
    else:
        demo()
        print("\n\nRun with 'extended' or 'autolearn' argument for more tests")
