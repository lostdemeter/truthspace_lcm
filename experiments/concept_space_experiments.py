#!/usr/bin/env python3
"""
Concept Space Experiments

Exploring how to work directly in concept space, including:
1. Loading distilled concepts
2. ConceptIdentity as a quaternion
3. Concept arithmetic (analogies)
4. Concept navigation
5. Concept clustering

Key insight: ConceptIdentity might be a quaternion with axes:
- Category (what it IS)
- Action (what it DOES)  
- Target (what it acts ON)
- Relation (what it connects to)

This is different from SemanticQuaternion which encodes:
- Gender, Age, Agency, Animacy

The two quaternions might be complementary:
- SemanticQuaternion: INTRINSIC properties (gender, age, animacy)
- IdentityQuaternion: RELATIONAL properties (category, action, target, relation)

Author: Lesley Gushurst
License: GPLv3
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.distilled_lcm import DistilledLCM
from truthspace_lcm.core.semantic_quaternion import SemanticQuaternion

PHI = 1.618034


@dataclass
class IdentityQuaternion:
    """
    A quaternion encoding a concept's IDENTITY in relational space.
    
    KEY INSIGHT: This is a DIFFERENT quaternion from SemanticQuaternion.
    
    SemanticQuaternion encodes INTRINSIC properties:
    - Gender, Age, Agency, Animacy
    
    IdentityQuaternion encodes RELATIONAL properties:
    - w: φ-direction (agency - from corpus)
    - x: Action signature (hash of top actions)
    - y: Target signature (hash of top targets)  
    - z: Category signature (hash of category)
    
    The signatures are computed by hashing the actual words into
    a position on the unit circle, giving each concept a unique
    "fingerprint" based on WHAT it does, not just HOW MUCH.
    
    This allows:
    - Concepts with same actions to cluster (detectives together)
    - Concepts with same targets to cluster (things that act on evidence)
    - Analogies to work (A:B :: C:? finds concepts with similar deltas)
    """
    w: float = 0.0  # φ-direction (agency)
    x: float = 0.0  # Action signature
    y: float = 0.0  # Target signature
    z: float = 0.0  # Category signature
    
    # Store the actual values for reference
    category: str = ""
    top_actions: List[str] = field(default_factory=list)
    top_targets: List[str] = field(default_factory=list)
    top_relations: List[str] = field(default_factory=list)
    
    def __add__(self, other: 'IdentityQuaternion') -> 'IdentityQuaternion':
        return IdentityQuaternion(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )
    
    def __sub__(self, other: 'IdentityQuaternion') -> 'IdentityQuaternion':
        return IdentityQuaternion(
            self.w - other.w,
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )
    
    def __mul__(self, scalar: float) -> 'IdentityQuaternion':
        return IdentityQuaternion(
            self.w * scalar,
            self.x * scalar,
            self.y * scalar,
            self.z * scalar
        )
    
    @property
    def magnitude(self) -> float:
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'IdentityQuaternion':
        mag = self.magnitude
        if mag == 0:
            return IdentityQuaternion(0, 0, 0, 1)
        return IdentityQuaternion(
            self.w / mag,
            self.x / mag,
            self.y / mag,
            self.z / mag
        )
    
    def dot(self, other: 'IdentityQuaternion') -> float:
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    
    def cosine_similarity(self, other: 'IdentityQuaternion') -> float:
        mag1 = self.magnitude
        mag2 = other.magnitude
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return self.dot(other) / (mag1 * mag2)
    
    def __repr__(self) -> str:
        return f"IQ(w={self.w:.2f}, x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


def word_to_angle(word: str) -> float:
    """
    Hash a word to an angle on the unit circle.
    
    This gives each word a unique "position" that can be used
    to create semantic signatures.
    """
    h = hash(word.lower()) % 10000
    return (h / 10000) * 2 * math.pi


def words_to_signature(words: List[str]) -> float:
    """
    Combine multiple words into a single signature.
    
    Uses vector addition on the unit circle - words that
    appear together will reinforce, random words will cancel.
    """
    if not words:
        return 0.0
    
    x_sum = sum(math.cos(word_to_angle(w)) for w in words)
    y_sum = sum(math.sin(word_to_angle(w)) for w in words)
    
    # Return the angle of the resultant vector
    return math.atan2(y_sum, x_sum) / math.pi  # Normalize to [-1, 1]


class ConceptSpace:
    """
    A space of concepts with quaternion representations.
    
    Each concept has TWO quaternion representations:
    1. SemanticQuaternion: Intrinsic properties (gender, age, agency, animacy)
    2. IdentityQuaternion: Relational properties (category, action, target, relation)
    
    Together they form an 8D concept space.
    """
    
    def __init__(self, distilled_path: str = "truthspace_lcm/concepts_distilled.json"):
        self.distilled = DistilledLCM()
        self.distilled.load(distilled_path)
        self.identity_cache: Dict[str, IdentityQuaternion] = {}
        
        # Category words for detecting what a concept IS
        self.category_words = {
            'detective', 'doctor', 'scientist', 'teacher', 'writer',
            'philosopher', 'artist', 'leader', 'hero', 'villain',
            'science', 'field', 'discipline', 'study', 'branch',
            'person', 'character', 'figure', 'companion', 'assistant',
            'physicist', 'chemist', 'biologist', 'mathematician',
            'author', 'poet', 'novelist', 'playwright',
            'king', 'queen', 'prince', 'princess', 'emperor',
            'theory', 'law', 'principle', 'concept', 'idea',
        }
    
    def get_identity(self, word: str) -> Optional[IdentityQuaternion]:
        """Get or compute the IdentityQuaternion for a concept."""
        word = word.lower()
        
        if word in self.identity_cache:
            return self.identity_cache[word]
        
        concept = self.distilled.get_concept(word)
        if not concept:
            return None
        
        # Extract semantic content
        top_actions = [a for a, _ in concept.actions[:5]]
        top_targets = [t for t, _ in concept.targets[:5]]
        
        # Detect category from targets
        category = ""
        for target, count in concept.targets[:5]:
            if target in self.category_words:
                category = target
                break
        
        # Get related concepts
        related = self.distilled.find_related(word, top_k=5)
        top_relations = [r for r, _ in related[:3]]
        
        # Compute quaternion components using SIGNATURES
        # w: φ-direction (agency) - directly from corpus
        w = concept.phi_direction
        
        # x: Action signature - hash of what this concept DOES
        x = words_to_signature(top_actions)
        
        # y: Target signature - hash of what this concept acts ON
        y = words_to_signature(top_targets)
        
        # z: Category signature - hash of what this concept IS
        z = words_to_signature([category] if category else [])
        
        iq = IdentityQuaternion(
            w=w,
            x=x,
            y=y,
            z=z,
            category=category,
            top_actions=top_actions,
            top_targets=top_targets,
            top_relations=top_relations,
        )
        
        self.identity_cache[word] = iq
        return iq
    
    def describe_identity(self, word: str) -> str:
        """Describe a concept's identity in natural language."""
        iq = self.get_identity(word)
        if not iq:
            return f"Unknown concept: {word}"
        
        parts = [f"**{word.title()}**"]
        
        if iq.category:
            parts.append(f"  Category: {iq.category}")
        
        if iq.top_actions:
            parts.append(f"  Actions: {', '.join(iq.top_actions)}")
        
        if iq.top_targets:
            parts.append(f"  Targets: {', '.join(iq.top_targets)}")
        
        if iq.top_relations:
            parts.append(f"  Related: {', '.join(iq.top_relations)}")
        
        parts.append(f"  Quaternion: {iq}")
        
        return "\n".join(parts)
    
    def similarity(self, word1: str, word2: str) -> float:
        """Compute similarity between two concepts using IdentityQuaternion."""
        iq1 = self.get_identity(word1)
        iq2 = self.get_identity(word2)
        
        if not iq1 or not iq2:
            return 0.0
        
        return iq1.cosine_similarity(iq2)
    
    def analogy(self, a: str, b: str, c: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Complete analogy: A is to B as C is to ?
        
        In CONCEPT SPACE, analogies work by finding PARALLEL RELATIONSHIPS.
        
        Key insight: If B is a category word (detective, doctor, science),
        then we're asking "what category is C?" - this is a CLASSIFICATION task.
        
        If B is not a category, we look for concepts that relate to C
        the same way B relates to A.
        """
        a, b, c = a.lower(), b.lower(), c.lower()
        
        concept_a = self.distilled.get_concept(a)
        concept_b = self.distilled.get_concept(b)
        concept_c = self.distilled.get_concept(c)
        
        if not all([concept_a, concept_c]):
            return []
        
        results = []
        
        # Case 1: B is a category word -> find C's category
        if b in self.category_words:
            # Look for category words in C's targets
            for target, count in concept_c.targets:
                if target in self.category_words:
                    results.append((target, count))
            
            # If no category found, infer from similar concepts
            if not results:
                # Find concepts similar to C that have categories
                c_related = self.distilled.find_related(c, top_k=20)
                for word, score in c_related:
                    related_concept = self.distilled.get_concept(word)
                    if related_concept:
                        for target, count in related_concept.targets:
                            if target in self.category_words:
                                results.append((target, score * count))
                                break
        
        # Case 2: B is in A's targets -> find what's in C's targets
        elif concept_b:
            a_targets = [t for t, _ in concept_a.targets]
            if b in a_targets:
                # A acts on B, find what C acts on
                for target, count in concept_c.targets:
                    results.append((target, count))
            else:
                # General case: find concepts related to C with similar φ-direction to B
                c_related = self.distilled.find_related(c, top_k=30)
                b_phi = concept_b.phi_direction
                
                for word, score in c_related:
                    if word in {a, b, c}:
                        continue
                    related_concept = self.distilled.get_concept(word)
                    if related_concept:
                        # Score by similarity in φ-direction
                        phi_diff = abs(related_concept.phi_direction - b_phi)
                        combined_score = score * (1 - phi_diff / 2)
                        results.append((word, combined_score))
        
        # Deduplicate and sort
        seen = set()
        unique_results = []
        for word, score in sorted(results, key=lambda x: -x[1]):
            if word not in seen and word not in {a, b, c}:
                seen.add(word)
                unique_results.append((word, score))
        
        return unique_results[:k]
    
    def cluster_by_category(self) -> Dict[str, List[str]]:
        """Group concepts by their detected category."""
        clusters = {}
        
        for word in self.distilled.concepts:
            iq = self.get_identity(word)
            if iq and iq.category:
                if iq.category not in clusters:
                    clusters[iq.category] = []
                clusters[iq.category].append(word)
        
        return clusters
    
    def find_by_profile(self, 
                        min_agency: float = -1.0,
                        max_agency: float = 1.0,
                        has_category: bool = False,
                        limit: int = 20) -> List[Tuple[str, IdentityQuaternion]]:
        """Find concepts matching a quaternion profile."""
        results = []
        
        for word in self.distilled.concepts:
            iq = self.get_identity(word)
            if not iq:
                continue
            
            if iq.w >= min_agency and iq.w <= max_agency:
                if has_category and not iq.category:
                    continue
                results.append((word, iq))
        
        # Sort by agency (w component)
        results.sort(key=lambda x: -x[1].w)
        return results[:limit]


def explore_dual_quaternion():
    """
    Explore the idea that concepts have TWO quaternion representations:
    
    1. SemanticQuaternion (intrinsic): Gender, Age, Agency, Animacy
       - Learned from word relationships (king-queen, man-woman)
       - Static properties of the concept itself
    
    2. IdentityQuaternion (relational): φ-direction, Actions, Targets, Category
       - Learned from corpus frames
       - Dynamic properties from how the concept is USED
    
    Together they form an 8D concept space.
    
    The key insight: ConceptIdentity IS a quaternion because it has 4 components:
    - WHAT it IS (category)
    - WHAT it DOES (actions)
    - WHAT it acts ON (targets)
    - HOW MUCH agency it has (φ-direction)
    
    This parallels the SemanticQuaternion's 4 components:
    - Gender (male/female/neutral)
    - Age (young/adult)
    - Agency (initiator/receiver)
    - Animacy (human/animal/thing)
    
    The two quaternions are ORTHOGONAL:
    - SemanticQuaternion: WHO/WHAT the concept is intrinsically
    - IdentityQuaternion: HOW the concept behaves in context
    """
    print("\n" + "=" * 70)
    print("DUAL QUATERNION EXPLORATION")
    print("=" * 70)
    
    from truthspace_lcm.core.semantic_quaternion import (
        SemanticQuaternion, 
        SemanticQuaternionNavigator,
        DEFAULT_SEMANTIC_FEATURES
    )
    from truthspace_lcm.core.distilled_lcm import DistilledLCM
    
    # Load distilled concepts
    distilled = DistilledLCM()
    distilled.load("truthspace_lcm/concepts_distilled.json")
    
    # Compare concepts that have both quaternion types
    test_concepts = ['holmes', 'watson', 'detective', 'doctor', 'king', 'queen', 'physics', 'darwin']
    
    print("\n--- Dual Quaternion Comparison ---")
    print(f"{'Concept':<12} {'Semantic (x,y,z,w)':<30} {'Identity (φ,actions,targets)':<40}")
    print("-" * 82)
    
    for word in test_concepts:
        # Get SemanticQuaternion (if defined)
        sq = DEFAULT_SEMANTIC_FEATURES.get(word)
        sq_str = f"({sq.x:.1f}, {sq.y:.1f}, {sq.z:.1f}, {sq.w:.1f})" if sq else "(not defined)"
        
        # Get IdentityQuaternion from corpus
        concept = distilled.get_concept(word)
        if concept:
            actions = [a for a, _ in concept.actions[:2]]
            targets = [t for t, _ in concept.targets[:2]]
            iq_str = f"φ={concept.phi_direction:.2f}, acts={actions}, tgts={targets}"
        else:
            iq_str = "(not in corpus)"
        
        print(f"{word:<12} {sq_str:<30} {iq_str:<40}")
    
    print("\n--- The Two Quaternions ---")
    print("""
    SemanticQuaternion (INTRINSIC):       IdentityQuaternion (RELATIONAL):
    ┌─────────────────────────────┐       ┌─────────────────────────────┐
    │ x: Gender (-1 F, +1 M)      │       │ w: φ-direction (agency)     │
    │ y: Age (-1 young, +1 adult) │       │ x: Action signature         │
    │ z: Agency (initiator/recv)  │       │ y: Target signature         │
    │ w: Animacy (human/thing)    │       │ z: Category signature       │
    └─────────────────────────────┘       └─────────────────────────────┘
    
    Together: 8D concept space
    
    Key insight: z (agency) in SemanticQuaternion ≈ w (φ-direction) in IdentityQuaternion
    They measure the same thing from different sources!
    """)
    
    # Test if φ-direction correlates with SemanticQuaternion.z
    print("\n--- Agency Correlation (SQ.z vs IQ.φ) ---")
    for word in ['holmes', 'watson', 'detective', 'doctor', 'king', 'queen']:
        sq = DEFAULT_SEMANTIC_FEATURES.get(word)
        concept = distilled.get_concept(word)
        
        if sq and concept:
            sq_agency = sq.z
            iq_agency = concept.phi_direction
            print(f"  {word}: SQ.z={sq_agency:.2f}, IQ.φ={iq_agency:.2f}, diff={abs(sq_agency - iq_agency):.2f}")


def demo():
    """Demonstrate concept space experiments."""
    print("=" * 70)
    print("CONCEPT SPACE EXPERIMENTS")
    print("=" * 70)
    
    space = ConceptSpace()
    
    # 1. Describe some concepts
    print("\n--- Concept Identities ---")
    for word in ['holmes', 'watson', 'physics', 'darwin', 'einstein']:
        print(space.describe_identity(word))
        print()
    
    # 2. Similarity
    print("\n--- Concept Similarity (IdentityQuaternion) ---")
    pairs = [
        ('holmes', 'watson'),
        ('holmes', 'moriarty'),
        ('physics', 'chemistry'),
        ('darwin', 'einstein'),
        ('holmes', 'physics'),
    ]
    for a, b in pairs:
        sim = space.similarity(a, b)
        print(f"  {a} <-> {b}: {sim:.3f}")
    
    # 3. Analogies
    print("\n--- Analogies (A:B :: C:?) ---")
    analogies = [
        ('holmes', 'detective', 'watson'),  # watson : ?
        ('physics', 'science', 'biology'),  # biology : ?
        ('king', 'queen', 'man'),           # man : ?
    ]
    for a, b, c in analogies:
        results = space.analogy(a, b, c, k=3)
        print(f"  {a}:{b} :: {c}:?")
        for word, score in results:
            print(f"    -> {word} ({score:.3f})")
    
    # 4. Category clusters
    print("\n--- Category Clusters ---")
    clusters = space.cluster_by_category()
    for category, words in sorted(clusters.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {category}: {len(words)} concepts")
        print(f"    Examples: {', '.join(words[:5])}")
    
    # 5. Find high-agency concepts (protagonists)
    print("\n--- High-Agency Concepts (Protagonists, w > 0.5) ---")
    protagonists = space.find_by_profile(min_agency=0.5, limit=10)
    for word, iq in protagonists:
        print(f"  {word}: {iq}")
    
    # 6. Find low-agency concepts (Objects/Themes)
    print("\n--- Low-Agency Concepts (Objects/Themes, w < -0.3) ---")
    objects = space.find_by_profile(max_agency=-0.3, limit=10)
    for word, iq in objects:
        print(f"  {word}: {iq}")
    
    # 7. Concepts with detected categories
    print("\n--- Concepts with Categories ---")
    categorized = space.find_by_profile(has_category=True, limit=15)
    for word, iq in categorized:
        print(f"  {word} ({iq.category}): {iq}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    explore_dual_quaternion()
    print("\n")
    demo()
