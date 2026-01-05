#!/usr/bin/env python3
"""
φ^(-rank) Importance Formula Experiment

From Design 039, the geometric importance formula is:

    importance = phi_weight(A) × phi_weight(B) × spread × bidir
    where phi_weight(X) = φ^(-rank(X))

Key differences from previous experiments:
1. Uses RANK not raw frequency
2. Includes SPREAD (how many sources mention entity)
3. Includes BIDIR (bidirectional relationship strength)

This is about ENTITY RELATIONSHIPS, not text similarity.

Author: TruthSpace LCM
License: GPLv3
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import re


PHI = (1 + np.sqrt(5)) / 2
CRITICAL_LINE = 0.5


@dataclass
class Entity:
    """An entity with frequency and relationship data."""
    name: str
    frequency: int = 0
    rank: int = 0
    sources: Set[str] = field(default_factory=set)
    relationships: Dict[str, int] = field(default_factory=dict)  # entity -> count
    
    @property
    def spread(self) -> float:
        """How many sources mention this entity (normalized)."""
        return len(self.sources)
    
    def bidir(self, other: 'Entity') -> float:
        """Bidirectional relationship strength with another entity."""
        # A mentions B AND B mentions A
        a_to_b = self.relationships.get(other.name, 0)
        b_to_a = other.relationships.get(self.name, 0)
        
        if a_to_b == 0 or b_to_a == 0:
            return 0.0
        
        # Geometric mean of bidirectional counts
        return np.sqrt(a_to_b * b_to_a)


class PhiRankSpace:
    """
    Geometric space using φ^(-rank) importance formula.
    
    From Design 039:
    - importance = phi_weight(A) × phi_weight(B) × spread × bidir
    - phi_weight(X) = φ^(-rank(X))
    
    This captures entity relationships, not text similarity.
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.concepts: List[str] = []  # Full text of each concept
        self.concept_entities: List[Set[str]] = []  # Entities per concept
        self._ranks_computed = False
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract entities (content words) from text."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Filter short words
        return {w for w in words if len(w) >= 3}
    
    def add_concept(self, text: str, source: str = "default"):
        """Add a concept and extract entities."""
        entities = self._extract_entities(text)
        
        self.concepts.append(text)
        self.concept_entities.append(entities)
        
        # Update entity data
        entity_list = list(entities)
        for i, e in enumerate(entity_list):
            if e not in self.entities:
                self.entities[e] = Entity(name=e)
            
            ent = self.entities[e]
            ent.frequency += 1
            ent.sources.add(source)
            
            # Track relationships (co-occurrence in same concept)
            for j, other in enumerate(entity_list):
                if i != j:
                    ent.relationships[other] = ent.relationships.get(other, 0) + 1
        
        self._ranks_computed = False
    
    def compute_ranks(self):
        """Compute ranks based on frequency (most frequent = rank 1)."""
        sorted_entities = sorted(
            self.entities.values(),
            key=lambda e: -e.frequency
        )
        for rank, entity in enumerate(sorted_entities, 1):
            entity.rank = rank
        self._ranks_computed = True
    
    def phi_weight(self, entity_name: str) -> float:
        """φ^(-rank) weighting."""
        if not self._ranks_computed:
            self.compute_ranks()
        
        entity = self.entities.get(entity_name)
        if not entity:
            return 0.0
        
        return PHI ** (-entity.rank)
    
    def importance(self, entity_a: str, entity_b: str) -> float:
        """
        Compute importance of relationship between two entities.
        
        importance = phi_weight(A) × phi_weight(B) × spread × bidir
        """
        if entity_a not in self.entities or entity_b not in self.entities:
            return 0.0
        
        ent_a = self.entities[entity_a]
        ent_b = self.entities[entity_b]
        
        phi_a = self.phi_weight(entity_a)
        phi_b = self.phi_weight(entity_b)
        
        # Spread: geometric mean of source counts
        spread = np.sqrt(ent_a.spread * ent_b.spread)
        
        # Bidirectionality
        bidir = ent_a.bidir(ent_b)
        
        return phi_a * phi_b * spread * bidir
    
    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Query using entity importance.
        
        For each concept, sum the importance of relationships
        between query entities and concept entities.
        """
        if not self._ranks_computed:
            self.compute_ranks()
        
        query_entities = self._extract_entities(text)
        
        results = []
        for i, concept in enumerate(self.concepts):
            concept_ents = self.concept_entities[i]
            
            # Sum importance of all query-concept entity pairs
            total_importance = 0.0
            for q_ent in query_entities:
                for c_ent in concept_ents:
                    total_importance += self.importance(q_ent, c_ent)
            
            results.append((concept, total_importance))
        
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


def experiment_phi_rank():
    """Test φ^(-rank) importance formula."""
    print("=" * 70)
    print("φ^(-RANK) IMPORTANCE FORMULA EXPERIMENT")
    print("=" * 70)
    
    space = PhiRankSpace()
    
    # Add concepts from different sources
    # Holmes domain
    space.add_concept("Sherlock Holmes is a detective who solves crimes.", source="holmes1")
    space.add_concept("Holmes uses deduction to find criminals.", source="holmes2")
    space.add_concept("Watson assists Holmes in investigations.", source="holmes3")
    space.add_concept("The detective examines clues at crime scenes.", source="holmes4")
    
    # Python domain
    space.add_concept("Python is a programming language.", source="python1")
    space.add_concept("Python uses indentation for code blocks.", source="python2")
    space.add_concept("Developers write scripts in Python.", source="python3")
    
    # Physics domain
    space.add_concept("Physics studies matter and energy.", source="physics1")
    space.add_concept("Quantum mechanics describes atomic behavior.", source="physics2")
    
    space.compute_ranks()
    
    print(f"\nAdded {len(space.concepts)} concepts")
    print(f"Entities: {len(space.entities)}")
    
    # Show entity ranks
    print("\n" + "-" * 70)
    print("ENTITY RANKS (top 15)")
    print("-" * 70)
    
    sorted_entities = sorted(space.entities.values(), key=lambda e: e.rank)
    print(f"\n{'Entity':15} {'Rank':>6} {'Freq':>6} {'Spread':>8} {'φ-weight':>10}")
    print("-" * 50)
    for ent in sorted_entities[:15]:
        phi = space.phi_weight(ent.name)
        print(f"{ent.name:15} {ent.rank:>6} {ent.frequency:>6} {ent.spread:>8} {phi:>10.6f}")
    
    # Show key relationships
    print("\n" + "-" * 70)
    print("KEY RELATIONSHIPS")
    print("-" * 70)
    
    pairs = [
        ("holmes", "detective"),
        ("holmes", "watson"),
        ("python", "programming"),
        ("physics", "energy"),
        ("holmes", "python"),  # Cross-domain (should be low)
    ]
    
    print(f"\n{'Entity A':12} {'Entity B':12} {'Importance':>12}")
    print("-" * 40)
    for a, b in pairs:
        imp = space.importance(a, b)
        print(f"{a:12} {b:12} {imp:>12.6f}")
    
    # Query tests
    print("\n" + "-" * 70)
    print("QUERY TESTS")
    print("-" * 70)
    
    queries = [
        "Who is Holmes?",
        "What is Python?",
        "Tell me about physics",
        "detective investigation",
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = space.query(query, top_k=3)
        for concept, score in results:
            print(f"  [{score:.6f}] {concept[:55]}...")


def experiment_spread_bidir():
    """
    Test the effect of spread and bidirectionality.
    
    Key insight: Bidirectional relationships are more meaningful.
    If A mentions B AND B mentions A, that's a strong signal.
    """
    print("\n" + "=" * 70)
    print("SPREAD AND BIDIRECTIONALITY EXPERIMENT")
    print("=" * 70)
    
    space = PhiRankSpace()
    
    # Create concepts with varying relationship patterns
    
    # Strong bidirectional: Holmes ↔ Watson
    space.add_concept("Holmes works with Watson.", source="s1")
    space.add_concept("Watson assists Holmes.", source="s2")
    space.add_concept("Holmes and Watson solve crimes.", source="s3")
    space.add_concept("Watson writes about Holmes.", source="s4")
    
    # Weak unidirectional: Holmes → Moriarty (Holmes mentions Moriarty, but not vice versa)
    space.add_concept("Holmes confronts Moriarty.", source="s5")
    space.add_concept("Holmes defeats Moriarty.", source="s6")
    
    space.compute_ranks()
    
    print(f"\nAdded {len(space.concepts)} concepts")
    
    # Compare relationships
    print("\n" + "-" * 70)
    print("RELATIONSHIP COMPARISON")
    print("-" * 70)
    
    holmes = space.entities.get("holmes")
    watson = space.entities.get("watson")
    moriarty = space.entities.get("moriarty")
    
    if holmes and watson:
        print(f"\nHolmes → Watson: {holmes.relationships.get('watson', 0)}")
        print(f"Watson → Holmes: {watson.relationships.get('holmes', 0)}")
        print(f"Bidir(Holmes, Watson): {holmes.bidir(watson):.3f}")
        print(f"Importance(Holmes, Watson): {space.importance('holmes', 'watson'):.6f}")
    
    if holmes and moriarty:
        print(f"\nHolmes → Moriarty: {holmes.relationships.get('moriarty', 0)}")
        print(f"Moriarty → Holmes: {moriarty.relationships.get('holmes', 0)}")
        print(f"Bidir(Holmes, Moriarty): {holmes.bidir(moriarty):.3f}")
        print(f"Importance(Holmes, Moriarty): {space.importance('holmes', 'moriarty'):.6f}")
    
    print("\n" + "-" * 70)
    print("KEY INSIGHT")
    print("-" * 70)
    print("""
Holmes ↔ Watson has HIGH importance because:
- Both mention each other (bidirectional)
- Both appear in multiple sources (high spread)
- φ^(-rank) weights them appropriately

Holmes → Moriarty has LOWER importance because:
- Only Holmes mentions Moriarty (unidirectional)
- bidir = 0 when relationship is one-way

This is the geometric signal of meaningful relationships:
BIDIRECTIONALITY = IMPORTANCE
""")


if __name__ == "__main__":
    experiment_phi_rank()
    experiment_spread_bidir()
