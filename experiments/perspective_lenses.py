#!/usr/bin/env python3
"""
Perspective Lenses for Truth Projection

Key insight from Watson example:
- Literally: Watson is a doctor (profession)
- Behaviorally: Watson is a companion (φ-direction shows he assists, accompanies)
- Relationally: Watson is Holmes's partner (targets show connection to Holmes)
- Narratively: Watson is the chronicler (role in story structure)

The SAME underlying truth, viewed through different lenses, yields different answers.

This is like how we project to English - the geometric truth needs a filter/perspective
to become a specific answer.

Author: Lesley Gushurst
License: GPLv3
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.distilled_lcm import DistilledLCM, Concept
from truthspace_lcm.core.semantic_quaternion import SemanticQuaternion, DEFAULT_SEMANTIC_FEATURES


class Lens(Enum):
    """Different perspectives for viewing truth."""
    LITERAL = "literal"         # What something IS (category, definition)
    BEHAVIORAL = "behavioral"   # How something ACTS (φ-direction, actions)
    RELATIONAL = "relational"   # How something CONNECTS (targets, relationships)
    NARRATIVE = "narrative"     # What ROLE something plays (story structure)
    INTRINSIC = "intrinsic"     # What something IS inherently (semantic quaternion)


@dataclass
class PerspectiveView:
    """A concept viewed through a specific lens."""
    concept: str
    lens: Lens
    primary_answer: str
    confidence: float
    supporting_evidence: List[str]
    
    def describe(self) -> str:
        return f"{self.concept} ({self.lens.value}): {self.primary_answer} (conf={self.confidence:.2f})"


class TruthPerspective:
    """
    View truth through different lenses.
    
    The geometric truth is the same, but the PERSPECTIVE determines
    what aspect we see.
    """
    
    def __init__(self, distilled_path: str = "truthspace_lcm/concepts_distilled.json"):
        self.distilled = DistilledLCM()
        self.distilled.load(distilled_path)
        
        # Category words for literal lens
        self.category_words = {
            'detective', 'doctor', 'scientist', 'teacher', 'writer',
            'philosopher', 'artist', 'leader', 'hero', 'villain',
            'companion', 'assistant', 'friend', 'enemy', 'partner',
            'physicist', 'biologist', 'chemist', 'mathematician',
            'chronicler', 'narrator', 'observer', 'participant',
        }
        
        # Role words for narrative lens
        self.role_words = {
            'protagonist', 'antagonist', 'sidekick', 'mentor', 'herald',
            'chronicler', 'narrator', 'observer', 'witness',
            'helper', 'trickster', 'guardian', 'shadow',
        }
        
        # Behavioral categories by φ-direction
        self.phi_categories = {
            (0.7, 1.0): ['initiator', 'leader', 'actor', 'agent'],
            (0.3, 0.7): ['participant', 'collaborator', 'contributor'],
            (-0.3, 0.3): ['neutral', 'observer', 'medium'],
            (-0.7, -0.3): ['recipient', 'affected', 'patient'],
            (-1.0, -0.7): ['receiver', 'object', 'target'],
        }
    
    def view_literal(self, word: str) -> Optional[PerspectiveView]:
        """
        Literal lens: What IS this thing? (category, definition)
        
        Looks for explicit category relationships in the corpus.
        """
        concept = self.distilled.get_concept(word.lower())
        if not concept:
            return None
        
        # Look for category words in targets (X is a Y)
        categories_found = []
        for target, count in concept.targets:
            if target in self.category_words:
                categories_found.append((target, count))
        
        # Also check related concepts for category hints
        related = self.distilled.find_related(word.lower(), top_k=10)
        for rel_word, score in related:
            if rel_word in self.category_words:
                categories_found.append((rel_word, score))
        
        if categories_found:
            # Sort by count/score
            categories_found.sort(key=lambda x: -x[1])
            primary = categories_found[0][0]
            confidence = min(1.0, categories_found[0][1] / 5)
        else:
            # No explicit category found
            primary = "(unknown)"
            confidence = 0.0
        
        return PerspectiveView(
            concept=word,
            lens=Lens.LITERAL,
            primary_answer=primary,
            confidence=confidence,
            supporting_evidence=[f"{cat}({count})" for cat, count in categories_found[:3]]
        )
    
    def view_behavioral(self, word: str) -> Optional[PerspectiveView]:
        """
        Behavioral lens: How does this thing ACT? (φ-direction, actions)
        
        Uses φ-direction to determine behavioral category.
        """
        concept = self.distilled.get_concept(word.lower())
        if not concept:
            return None
        
        phi = concept.phi_direction
        
        # Find behavioral category
        behavioral_type = "neutral"
        for (low, high), categories in self.phi_categories.items():
            if low <= phi <= high:
                behavioral_type = categories[0]
                break
        
        # Get primary actions
        actions = [act for act, _ in concept.actions[:5]]
        
        # Confidence based on how extreme the φ is
        confidence = abs(phi)
        
        return PerspectiveView(
            concept=word,
            lens=Lens.BEHAVIORAL,
            primary_answer=behavioral_type,
            confidence=confidence,
            supporting_evidence=[f"φ={phi:.2f}"] + actions[:3]
        )
    
    def view_relational(self, word: str) -> Optional[PerspectiveView]:
        """
        Relational lens: How does this thing CONNECT? (targets, relationships)
        
        Looks at what the concept acts on and relates to.
        """
        concept = self.distilled.get_concept(word.lower())
        if not concept:
            return None
        
        # Primary relationships from targets
        targets = [t for t, _ in concept.targets[:5]]
        
        # Find the most significant relationship
        if targets:
            primary_relation = f"connected to {targets[0]}"
            confidence = min(1.0, len(targets) / 5)
        else:
            primary_relation = "(isolated)"
            confidence = 0.0
        
        # Get related concepts
        related = self.distilled.find_related(word.lower(), top_k=5)
        related_words = [w for w, _ in related]
        
        return PerspectiveView(
            concept=word,
            lens=Lens.RELATIONAL,
            primary_answer=primary_relation,
            confidence=confidence,
            supporting_evidence=targets[:3] + related_words[:2]
        )
    
    def view_narrative(self, word: str) -> Optional[PerspectiveView]:
        """
        Narrative lens: What ROLE does this play? (story structure)
        
        Infers narrative role from behavioral and relational patterns.
        """
        concept = self.distilled.get_concept(word.lower())
        if not concept:
            return None
        
        phi = concept.phi_direction
        actions = [act for act, _ in concept.actions[:5]]
        targets = [t for t, _ in concept.targets[:5]]
        
        # Infer narrative role from patterns
        role = "participant"  # default
        confidence = 0.5
        evidence = []
        
        # Check for chronicler/narrator patterns
        chronicler_actions = {'write', 'record', 'document', 'narrate', 'describe', 'tell', 'chronicle'}
        if any(act in chronicler_actions for act in actions):
            role = "chronicler"
            confidence = 0.8
            evidence.append("writes/records")
        
        # Check for helper/sidekick patterns
        helper_actions = {'assist', 'help', 'support', 'accompany', 'follow', 'aid'}
        if any(act in helper_actions for act in actions):
            role = "helper/sidekick"
            confidence = 0.8
            evidence.append("assists/accompanies")
        
        # Check for protagonist patterns (high agency, central)
        if phi > 0.8 and len(targets) > 3:
            role = "protagonist"
            confidence = 0.7
            evidence.append(f"high agency (φ={phi:.2f})")
        
        # Check for antagonist patterns
        antagonist_actions = {'oppose', 'fight', 'attack', 'threaten', 'scheme'}
        if any(act in antagonist_actions for act in actions):
            role = "antagonist"
            confidence = 0.7
            evidence.append("opposes/threatens")
        
        evidence.append(f"φ={phi:.2f}")
        
        return PerspectiveView(
            concept=word,
            lens=Lens.NARRATIVE,
            primary_answer=role,
            confidence=confidence,
            supporting_evidence=evidence
        )
    
    def view_intrinsic(self, word: str) -> Optional[PerspectiveView]:
        """
        Intrinsic lens: What IS this thing inherently? (semantic quaternion)
        
        Uses predefined semantic properties: gender, age, agency, animacy.
        """
        word_lower = word.lower()
        
        if word_lower not in DEFAULT_SEMANTIC_FEATURES:
            return PerspectiveView(
                concept=word,
                lens=Lens.INTRINSIC,
                primary_answer="(not defined)",
                confidence=0.0,
                supporting_evidence=[]
            )
        
        sq = DEFAULT_SEMANTIC_FEATURES[word_lower]
        
        # Build description from quaternion
        properties = []
        if sq.x > 0.5:
            properties.append("male")
        elif sq.x < -0.5:
            properties.append("female")
        
        if sq.y > 0.5:
            properties.append("adult")
        elif sq.y < -0.5:
            properties.append("young")
        
        if sq.z > 0.5:
            properties.append("high-agency")
        elif sq.z < -0.5:
            properties.append("low-agency")
        
        if sq.w > 0.5:
            properties.append("human")
        elif sq.w < -0.5:
            properties.append("abstract")
        
        primary = ", ".join(properties) if properties else "neutral"
        
        return PerspectiveView(
            concept=word,
            lens=Lens.INTRINSIC,
            primary_answer=primary,
            confidence=1.0,  # Predefined, so high confidence
            supporting_evidence=[f"x={sq.x:.1f}", f"y={sq.y:.1f}", f"z={sq.z:.1f}", f"w={sq.w:.1f}"]
        )
    
    def view_all(self, word: str) -> Dict[Lens, PerspectiveView]:
        """View a concept through all lenses."""
        return {
            Lens.LITERAL: self.view_literal(word),
            Lens.BEHAVIORAL: self.view_behavioral(word),
            Lens.RELATIONAL: self.view_relational(word),
            Lens.NARRATIVE: self.view_narrative(word),
            Lens.INTRINSIC: self.view_intrinsic(word),
        }
    
    def compare_perspectives(self, word: str) -> None:
        """Print a comparison of all perspectives on a concept."""
        views = self.view_all(word)
        
        print(f"\n{'='*60}")
        print(f"PERSPECTIVES ON: {word.upper()}")
        print(f"{'='*60}")
        
        for lens, view in views.items():
            if view:
                print(f"\n{lens.value.upper()} LENS:")
                print(f"  Answer: {view.primary_answer}")
                print(f"  Confidence: {view.confidence:.2f}")
                print(f"  Evidence: {', '.join(view.supporting_evidence)}")
            else:
                print(f"\n{lens.value.upper()} LENS: (concept not found)")


def demo():
    """Demonstrate perspective lenses."""
    print("=" * 70)
    print("PERSPECTIVE LENSES FOR TRUTH PROJECTION")
    print("=" * 70)
    
    tp = TruthPerspective()
    
    # The Watson example - literally a doctor, behaviorally a companion
    print("\n" + "=" * 70)
    print("THE WATSON PARADOX")
    print("=" * 70)
    print("""
    Watson is LITERALLY a doctor (that's his profession in the stories).
    But BEHAVIORALLY, he acts as a companion/assistant (φ=0.73, assists Holmes).
    RELATIONALLY, he's connected to Holmes (his primary target).
    NARRATIVELY, he's the chronicler (he writes the stories).
    
    The SAME truth, viewed through different lenses:
    """)
    
    tp.compare_perspectives("watson")
    
    # Compare with Holmes
    tp.compare_perspectives("holmes")
    
    # Compare with a concept that's more consistent across lenses
    tp.compare_perspectives("detective")
    
    # The key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
    The geometric truth (φ-direction, actions, targets) captures
    BEHAVIORAL reality, not LITERAL definitions.
    
    When we ask "what is Watson?":
    - LITERAL lens: "doctor" (but not in our corpus!)
    - BEHAVIORAL lens: "initiator/participant" (φ=0.73)
    - RELATIONAL lens: "connected to Holmes"
    - NARRATIVE lens: "helper/sidekick" or "chronicler"
    
    The "correct" answer depends on which LENS you're using.
    
    This is like projecting to English - the same geometric truth
    can be expressed differently depending on perspective.
    """)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo()
