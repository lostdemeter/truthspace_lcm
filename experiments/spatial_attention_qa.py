#!/usr/bin/env python3
"""
Spatial Attention for Q&A: Finding What's IMPORTANT, Not Just Frequent

The Problem:
    "Holmes is a character who spoke often involving jabez"
    
    Jabez Wilson appears in ONE story with high local frequency.
    Watson appears across ALL stories - he's the DEFINING relationship.
    
    Frequency ≠ Importance

The Solution: Spatial Attention
    Like attention in transformers, we need to weight entities by:
    1. SPREAD - appears across many contexts (books/stories)
    2. RECIPROCITY - bidirectional relationship
    3. CENTRALITY - connected to other important entities
    
    importance(entity) = spread × reciprocity × centrality

This is the "path of spatial relativity" - tracing which entities are
truly central to understanding a character, not just locally frequent.
"""

import json
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SpatialAttention:
    """
    Compute importance weights for entities using spatial attention.
    
    This is analogous to attention in transformers:
    - Query: What is important about entity X?
    - Keys: All related entities
    - Values: Entity descriptions
    - Attention weights: spread × reciprocity × centrality
    """
    
    def __init__(self, frames: List[Dict]):
        self.frames = frames
        
        # Precompute entity statistics
        self._compute_entity_sources()
        self._compute_co_occurrence_graph()
        self._compute_bidirectional_relations()
    
    def _compute_entity_sources(self):
        """Compute which sources each entity appears in."""
        self.entity_sources: Dict[str, Set[str]] = defaultdict(set)
        
        for f in self.frames:
            text = f.get('text', '').lower()
            source = f.get('source', 'unknown')
            agent = f.get('agent', '')
            patient = f.get('patient', '')
            
            if agent:
                self.entity_sources[agent].add(source)
            if patient:
                self.entity_sources[patient].add(source)
    
    def _compute_co_occurrence_graph(self):
        """Build entity co-occurrence graph."""
        self.co_occurrence: Dict[str, Counter] = defaultdict(Counter)
        
        for f in self.frames:
            text = f.get('text', '').lower()
            agent = f.get('agent', '')
            patient = f.get('patient', '')
            
            # Direct agent-patient relationship
            if agent and patient and agent != patient:
                self.co_occurrence[agent][patient] += 1
                self.co_occurrence[patient][agent] += 1
    
    def _compute_bidirectional_relations(self):
        """Compute bidirectional relationship strength."""
        self.bidirectional: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(dict)
        
        for f in self.frames:
            agent = f.get('agent', '')
            patient = f.get('patient', '')
            
            if agent and patient and agent != patient:
                if patient not in self.bidirectional[agent]:
                    self.bidirectional[agent][patient] = [0, 0]
                self.bidirectional[agent][patient][0] += 1  # agent → patient
                
                if agent not in self.bidirectional[patient]:
                    self.bidirectional[patient][agent] = [0, 0]
                self.bidirectional[patient][agent][1] += 1  # patient ← agent
    
    def spread_score(self, entity: str) -> float:
        """
        SPREAD: How many different sources does this entity appear in?
        
        High spread = appears across many books/stories = more important
        """
        sources = self.entity_sources.get(entity, set())
        total_sources = len(set(f.get('source', '') for f in self.frames))
        
        if total_sources == 0:
            return 0.0
        
        return len(sources) / total_sources
    
    def reciprocity_score(self, entity1: str, entity2: str) -> float:
        """
        RECIPROCITY: Is the relationship bidirectional?
        
        High reciprocity = both entities reference each other = stronger bond
        """
        if entity2 not in self.bidirectional.get(entity1, {}):
            return 0.0
        
        outgoing, incoming = self.bidirectional[entity1][entity2]
        total = outgoing + incoming
        
        if total == 0:
            return 0.0
        
        # Reciprocity is higher when both directions are present
        balance = min(outgoing, incoming) / max(outgoing, incoming) if max(outgoing, incoming) > 0 else 0
        magnitude = np.log1p(total)  # Log scale for magnitude
        
        return balance * magnitude
    
    def centrality_score(self, entity: str) -> float:
        """
        CENTRALITY: How connected is this entity to other entities?
        
        High centrality = connected to many other entities = more important
        """
        connections = self.co_occurrence.get(entity, Counter())
        
        if not connections:
            return 0.0
        
        # Number of unique connections
        num_connections = len(connections)
        
        # Total connection strength
        total_strength = sum(connections.values())
        
        # Normalize by max possible
        max_entities = len(self.entity_sources)
        
        return (num_connections / max_entities) * np.log1p(total_strength)
    
    def importance_score(self, query_entity: str, related_entity: str) -> float:
        """
        Compute overall importance of related_entity to query_entity.
        
        importance = spread × reciprocity × centrality
        """
        spread = self.spread_score(related_entity)
        reciprocity = self.reciprocity_score(query_entity, related_entity)
        centrality = self.centrality_score(related_entity)
        
        # Combine with geometric mean (all factors matter)
        # Add small epsilon to avoid zero
        eps = 0.01
        score = (spread + eps) * (reciprocity + eps) * (centrality + eps)
        
        return score
    
    def get_important_relations(self, entity: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Get the k most important entities related to the query entity.
        
        This is the ATTENTION operation - finding what matters most.
        """
        candidates = self.co_occurrence.get(entity, Counter())
        
        if not candidates:
            return []
        
        # Score each candidate
        scored = []
        for related, freq in candidates.items():
            importance = self.importance_score(entity, related)
            scored.append((related, importance, freq))
        
        # Sort by importance (not frequency!)
        scored.sort(key=lambda x: -x[1])
        
        return [(e, score) for e, score, _ in scored[:k]]


def run_experiment():
    """Test spatial attention on Holmes corpus."""
    
    print("=" * 70)
    print("SPATIAL ATTENTION EXPERIMENT")
    print("=" * 70)
    print()
    
    # Load corpus
    with open('truthspace_lcm/concept_corpus.json', 'r') as f:
        corpus = json.load(f)
    
    attention = SpatialAttention(corpus['frames'])
    
    # Test on Holmes
    print("## HOLMES: Important Relations (with Spatial Attention)")
    print()
    
    important = attention.get_important_relations('holmes', k=10)
    
    print("Entity          | Importance | Spread | Reciprocity | Centrality")
    print("-" * 70)
    
    for entity, importance in important:
        spread = attention.spread_score(entity)
        recip = attention.reciprocity_score('holmes', entity)
        central = attention.centrality_score(entity)
        print(f"{entity:15} | {importance:10.4f} | {spread:6.3f} | {recip:11.3f} | {central:10.3f}")
    
    print()
    print("## COMPARISON: Frequency vs Importance")
    print()
    
    # Get frequency-based ranking
    freq_ranking = attention.co_occurrence.get('holmes', Counter()).most_common(10)
    
    print("By FREQUENCY (current system):")
    for entity, freq in freq_ranking:
        print(f"  {entity}: {freq}")
    
    print()
    print("By IMPORTANCE (spatial attention):")
    for entity, importance in important:
        print(f"  {entity}: {importance:.4f}")
    
    print()
    print("=" * 70)
    print("## WATSON vs JABEZ: Why Attention Matters")
    print("=" * 70)
    print()
    
    # Detailed comparison
    for entity in ['watson', 'jabez', 'lestrade']:
        print(f"{entity.upper()}:")
        print(f"  Spread: {attention.spread_score(entity):.3f} (appears in {len(attention.entity_sources.get(entity, set()))} sources)")
        print(f"  Reciprocity with Holmes: {attention.reciprocity_score('holmes', entity):.3f}")
        print(f"  Centrality: {attention.centrality_score(entity):.3f}")
        print(f"  IMPORTANCE: {attention.importance_score('holmes', entity):.4f}")
        print()
    
    print("=" * 70)
    print("## GENERATING ATTENTION-WEIGHTED ANSWER")
    print("=" * 70)
    print()
    
    # Get top important relation
    if important:
        top_relation = important[0][0]
        print(f"Most important relation for Holmes: {top_relation}")
        print()
        
        # Build better answer
        print("OLD ANSWER (frequency-based):")
        print("  Holmes is a character from Sherlock Holmes who spoke often involving jabez")
        print()
        print("NEW ANSWER (attention-weighted):")
        print(f"  Holmes is a character from Sherlock Holmes who spoke often involving {top_relation}")
    
    return attention


def analyze_attention_geometry():
    """Analyze the geometric structure of attention weights."""
    
    print()
    print("=" * 70)
    print("GEOMETRIC ANALYSIS: Attention as Spatial Path")
    print("=" * 70)
    print()
    
    # Load corpus
    with open('truthspace_lcm/concept_corpus.json', 'r') as f:
        corpus = json.load(f)
    
    attention = SpatialAttention(corpus['frames'])
    
    # The key insight: attention weights form a GEOMETRY
    # - High spread = entity is "everywhere" in the space
    # - High reciprocity = strong bidirectional connection
    # - High centrality = hub in the network
    
    print("The attention weights define a GEOMETRY:")
    print()
    print("  SPREAD = how 'distributed' an entity is across the space")
    print("  RECIPROCITY = strength of bidirectional path")
    print("  CENTRALITY = how many paths pass through this entity")
    print()
    print("This is like finding the 'center of mass' of relationships!")
    print()
    
    # Visualize as distances
    print("## ATTENTION AS DISTANCE")
    print()
    print("If we think of importance as inverse distance:")
    print("  distance(Holmes, Watson) = 1/importance ≈ close")
    print("  distance(Holmes, Jabez) = 1/importance ≈ far")
    print()
    
    for entity in ['watson', 'jabez', 'lestrade', 'moriarty']:
        imp = attention.importance_score('holmes', entity)
        if imp > 0:
            dist = 1.0 / imp
            print(f"  Holmes → {entity}: importance={imp:.4f}, distance={dist:.2f}")
    
    print()
    print("=" * 70)
    print("KEY INSIGHT: Attention IS Geometry")
    print("=" * 70)
    print()
    print("The attention mechanism defines a METRIC SPACE where:")
    print("  - Important entities are CLOSE to the query")
    print("  - Unimportant entities are FAR from the query")
    print()
    print("This is the 'path of spatial relativity' - the geometry")
    print("of relationships that determines what's truly important.")


if __name__ == "__main__":
    attention = run_experiment()
    analyze_attention_geometry()
