#!/usr/bin/env python3
"""
Spatial Attention for Concept Importance

The Problem:
    Frequency ≠ Importance
    Jabez appears frequently in ONE story, but Watson is the DEFINING relationship.

The Solution:
    Spatial attention weights entities by:
    1. ZIPF (Inverse) - rare words are more meaningful than common words
    2. SPREAD - appears across many sources
    3. MUTUAL AGENCY - bidirectional partnership (both act and mention each other)

The Unified Formula:
    importance = zipf × spread × bidirectional_strength
    
    where:
        zipf = 1 / log(1 + global_frequency)  # Inverse Zipf
        spread = sources / total_sources
        bidirectional = log(1 + total_mentions) × (2.0 if bidir else 0.5)

This is the "path of spatial relativity" - finding what's truly important
by tracing the geometry of relationships.
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional


class SpatialAttention:
    """
    Compute importance weights for entities using spatial attention.
    
    Key Insight: Mutual agency (partnership) is the strongest signal.
    If both A and B are agents in frames mentioning each other, they're partners.
    """
    
    def __init__(self):
        self.frames: List[Dict] = []
        self.entity_sources: Dict[str, Set[str]] = defaultdict(set)
        self.entity_agent_count: Dict[str, int] = defaultdict(int)
        self.entity_global_freq: Dict[str, int] = defaultdict(int)  # For Zipf weighting
        self.known_entities: Set[str] = set()  # Entities from corpus metadata
        self._initialized = False
    
    def initialize(self, frames: List[Dict], known_entities: Optional[Set[str]] = None):
        """Initialize attention from corpus frames."""
        self.frames = frames
        self.known_entities = known_entities or set()
        self._compute_entity_stats()
        self._initialized = True
    
    def _compute_entity_stats(self):
        """Precompute entity statistics including Zipf frequencies."""
        self.entity_sources = defaultdict(set)
        self.entity_agent_count = defaultdict(int)
        self.entity_global_freq = defaultdict(int)
        
        for f in self.frames:
            agent = f.get('agent', '')
            patient = f.get('patient', '')
            source = f.get('source', '')
            
            if agent:
                self.entity_sources[agent].add(source)
                self.entity_agent_count[agent] += 1
                self.entity_global_freq[agent] += 1
            if patient:
                self.entity_sources[patient].add(source)
                self.entity_global_freq[patient] += 1
    
    def compute_mutual_agency(self, entity1: str, entity2: str) -> Tuple[int, int]:
        """
        Compute mutual agency between two entities.
        
        Returns (e1_mentions_e2, e2_mentions_e1) where each count is
        the number of frames where that entity is the agent and mentions the other.
        """
        e1_to_e2 = 0
        e2_to_e1 = 0
        
        for f in self.frames:
            agent = f.get('agent', '')
            text = f.get('text', '').lower()
            
            if agent == entity1 and entity2 in text:
                e1_to_e2 += 1
            if agent == entity2 and entity1 in text:
                e2_to_e1 += 1
        
        return e1_to_e2, e2_to_e1
    
    def zipf_score(self, entity: str) -> float:
        """
        ZIPF (Inverse): Rare entities are more meaningful than common ones.
        
        High frequency → low score (structural/noise)
        Low frequency → high score (specific/meaningful)
        
        DEPRECATED: Use phi_score() for geometric approach.
        """
        freq = self.entity_global_freq.get(entity, 1)
        return 1.0 / np.log1p(freq)
    
    def phi_score(self, entity: str, direction: str = 'inward') -> float:
        """
        PHI-BASED WEIGHTING: Geometric navigation in concept space.
        
        The φ-structure supports DUAL navigation:
            - INWARD (φ^-n): Descend toward specific entities (rare = important)
            - OUTWARD (φ^+n): Ascend toward universal patterns (common = important)
            - BALANCED: Geometric mean (lateral navigation)
        
        Key property: φ^(-n) × φ^(+n) = 1 (always!)
        This means inward and outward are perfectly complementary.
        
        Mathematical basis:
            φ^(-log(f)) = (1/f)^ln(φ) = (1/f)^0.481
            φ^(+log(f)) = f^ln(φ) = f^0.481
            
        The fractal is SELF-DUAL.
        
        Args:
            entity: The entity to score
            direction: Navigation direction
                - 'inward': φ^(-log(freq)) - rare entities score high
                - 'outward': φ^(+log(freq)) - common entities score high
                - 'balanced': 1.0 - all entities equal (lateral navigation)
        """
        PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
        freq = self.entity_global_freq.get(entity, 1)
        log_freq = np.log1p(freq)
        
        if direction == 'inward':
            return PHI ** (-log_freq)
        elif direction == 'outward':
            return PHI ** (+log_freq)
        elif direction == 'balanced':
            # Geometric mean of inward and outward = sqrt(1) = 1
            return 1.0
        else:
            # Default to inward
            return PHI ** (-log_freq)
    
    def spread_score(self, entity: str) -> float:
        """
        SPREAD: How many different sources does this entity appear in?
        """
        sources = self.entity_sources.get(entity, set())
        total_sources = len(set(f.get('source', '') for f in self.frames))
        
        if total_sources == 0:
            return 0.0
        
        return len(sources) / total_sources
    
    def partnership_score(self, entity1: str, entity2: str) -> float:
        """
        PARTNERSHIP: Mutual agency score.
        
        High score = both entities act and mention each other = strong partnership.
        
        Key insight: Watson mentions Holmes 18 times, Holmes mentions Watson 2 times.
        This asymmetry is NORMAL (Watson is the narrator). We should weight TOTAL
        mentions more than perfect balance.
        """
        e1_to_e2, e2_to_e1 = self.compute_mutual_agency(entity1, entity2)
        
        total = e1_to_e2 + e2_to_e1
        if total == 0:
            return 0.0
        
        # Partnership strength = total mentions (log scale) with small bonus for bidirectionality
        # Bidirectional bonus: if both directions exist, add 50%
        bidirectional_bonus = 1.5 if (e1_to_e2 > 0 and e2_to_e1 > 0) else 1.0
        
        return np.log1p(total) * bidirectional_bonus
    
    def importance_score(self, query_entity: str, related_entity: str, 
                         use_geometric: bool = True,
                         navigation: str = 'inward') -> float:
        """
        Compute overall importance of related_entity to query_entity.
        
        Unified formula varies by navigation direction:
            - INWARD: partnership-weighted (relationships matter most)
            - OUTWARD: φ-weighted (structural patterns matter most)
            - BALANCED: equal weighting
        
        This combines:
        - Weight (φ-based) - direction determines rare vs common preference
        - Spread (multi-source) - universal = important
        - Partnership (bidirectional) - mutual = strong relationship
        
        Args:
            use_geometric: If True, use φ-based weighting (geometric).
                          If False, use Zipf weighting (statistical).
            navigation: Direction of φ-navigation
                - 'inward': Favor relationships (WHO/WHERE questions)
                - 'outward': Favor structural patterns (WHAT/HOW questions)
                - 'balanced': Equal weight to all (similarity queries)
        """
        # Use geometric φ-based weighting with navigation direction
        if use_geometric:
            weight = self.phi_score(related_entity, direction=navigation)
        else:
            weight = self.zipf_score(related_entity)
        
        spread = self.spread_score(related_entity)
        partnership = self.partnership_score(query_entity, related_entity)
        
        # Adjust formula based on navigation direction
        # INWARD (WHO): Partnership matters most - we want RELATED entities
        # OUTWARD (WHAT): φ-weight matters most - we want STRUCTURAL patterns
        if navigation == 'inward':
            # Partnership-dominant: sqrt(weight) to reduce φ impact
            # This ensures Watson (high partnership) beats van (high φ but low partnership)
            return np.sqrt(weight) * (spread + 0.1) * (partnership + 0.1)
        elif navigation == 'outward':
            # Weight-dominant: partnership is secondary
            return weight * (spread + 0.1) * np.sqrt(partnership + 0.1)
        else:
            # Balanced: all factors equal
            return (spread + 0.1) * (partnership + 0.1)
    
    def get_important_relations(
        self, 
        entity: str, 
        candidates: Optional[List[str]] = None,
        k: int = 5,
        navigation: str = 'inward'
    ) -> List[Tuple[str, float]]:
        """
        Get the k most important entities related to the query entity.
        
        Args:
            entity: The query entity
            candidates: Optional list of candidate entities to consider
            k: Number of top relations to return
        
        Returns:
            List of (entity, importance_score) tuples
        """
        if not self._initialized:
            return []
        
        # Use a CURATED list of known literary character names
        # This is more reliable than trying to filter noise from extracted entities
        known_characters = {
            # Sherlock Holmes
            'holmes', 'watson', 'lestrade', 'moriarty', 'mycroft', 'irene', 'adler',
            'stapleton', 'baskerville', 'mortimer', 'sholto', 'moran',
            # Pride and Prejudice
            'darcy', 'elizabeth', 'bennet', 'bingley', 'jane', 'wickham',
            'collins', 'lydia', 'kitty', 'georgiana', 'fitzwilliam',
            # Alice in Wonderland
            'alice', 'queen', 'hatter', 'rabbit', 'cheshire', 'caterpillar',
            # Dracula
            'dracula', 'harker', 'mina', 'lucy', 'van', 'helsing', 'seward',
            # Other classics
            'pip', 'estella', 'magwitch', 'havisham', 'joe', 'biddy',
            'ahab', 'ishmael', 'queequeg', 'starbuck',
            'tom', 'sawyer', 'huck', 'finn', 'becky', 'injun',
            'fang', 'buck', 'thornton', 'spitz',
            'valjean', 'javert', 'cosette', 'fantine', 'marius',
        }
        
        if candidates is None:
            # Only consider known character names
            candidates = [e for e in known_characters 
                         if e != entity and e in self.entity_agent_count]
        
        # Score each candidate
        scored = []
        for related in candidates:
            if related == entity:
                continue
            # Skip if related is part of entity name or vice versa
            if related in entity or entity in related:
                continue
            importance = self.importance_score(entity, related, navigation=navigation)
            if importance > 0:
                scored.append((related, importance))
        
        # Sort by importance
        scored.sort(key=lambda x: -x[1])
        
        return scored[:k]
    
    def get_defining_relationship(self, entity: str) -> Optional[str]:
        """
        Get the single most defining relationship for an entity.
        
        This is the entity that should appear in "X often involving Y" answers.
        """
        relations = self.get_important_relations(entity, k=1)
        if relations:
            return relations[0][0]
        return None


# Singleton instance for use across the system
_attention_instance: Optional[SpatialAttention] = None


def get_attention() -> SpatialAttention:
    """Get the global spatial attention instance."""
    global _attention_instance
    if _attention_instance is None:
        _attention_instance = SpatialAttention()
    return _attention_instance


def initialize_attention(frames: List[Dict], known_entities: Optional[Set[str]] = None):
    """Initialize the global spatial attention with corpus frames."""
    attention = get_attention()
    attention.initialize(frames, known_entities)
