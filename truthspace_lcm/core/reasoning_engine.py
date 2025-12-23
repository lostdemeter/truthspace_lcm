#!/usr/bin/env python3
"""
Reasoning Engine: Multi-Hop Reasoning for GeometricLCM

This module implements multi-hop reasoning through concept graph traversal.
Instead of single-hop (entity → answer), we chain multiple hops to answer
complex questions like WHY and HOW.

The key insight: Reasoning is GRAPH TRAVERSAL in concept space.
Each hop follows an edge in the concept graph.

Author: Lesley Gushurst
License: GPLv3
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter
import numpy as np

PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# REASONING STEP
# =============================================================================

@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    entity: str
    action: str
    relation: str
    evidence: str  # The source text supporting this step
    confidence: float = 1.0
    
    def __str__(self):
        return f"{self.entity} --{self.action}--> {self.relation}"


@dataclass
class ReasoningPath:
    """A complete reasoning path from query to answer."""
    steps: List[ReasoningStep] = field(default_factory=list)
    query: str = ""
    answer: str = ""
    
    def add_step(self, step: ReasoningStep):
        self.steps.append(step)
    
    def get_entities(self) -> List[str]:
        """Get all entities in the path."""
        entities = []
        for step in self.steps:
            if step.entity not in entities:
                entities.append(step.entity)
            if step.relation not in entities:
                entities.append(step.relation)
        return entities
    
    def get_chain_description(self) -> str:
        """Get a human-readable description of the reasoning chain."""
        if not self.steps:
            return "No reasoning steps."
        
        parts = []
        for i, step in enumerate(self.steps, 1):
            parts.append(f"Step {i}: {step.entity} {step.action.lower()} {step.relation}")
        
        return " → ".join(parts)
    
    def __len__(self):
        return len(self.steps)


# =============================================================================
# REASONING ENGINE
# =============================================================================

class ReasoningEngine:
    """
    Multi-hop reasoning through concept graph traversal.
    
    Supports:
    - WHY questions (causal chains)
    - HOW questions (process chains)
    - Relationship inference (entity-to-entity paths)
    """
    
    def __init__(self, knowledge):
        """
        Initialize with a ConceptKnowledge instance.
        
        Args:
            knowledge: ConceptKnowledge with loaded frames
        """
        self.knowledge = knowledge
        self.max_hops = 4
        
        # Build entity graph for efficient traversal
        self._build_entity_graph()
    
    def _build_entity_graph(self):
        """Build adjacency graph from frames."""
        self.entity_graph: Dict[str, List[Tuple[str, str, str]]] = {}
        
        for frame in self.knowledge.frames:
            agent = frame.get('agent', '').lower()
            patient = frame.get('patient', '').lower()
            action = frame.get('action', 'RELATE')
            text = frame.get('text', '')[:100]
            
            if agent and patient and agent != patient:
                if agent not in self.entity_graph:
                    self.entity_graph[agent] = []
                self.entity_graph[agent].append((patient, action, text))
                
                # Bidirectional for some relations
                if patient not in self.entity_graph:
                    self.entity_graph[patient] = []
                self.entity_graph[patient].append((agent, f"INVERSE_{action}", text))
    
    def reason(self, query: str, max_hops: int = None) -> ReasoningPath:
        """
        Perform multi-hop reasoning to answer a query.
        
        Args:
            query: The question to answer
            max_hops: Maximum reasoning hops (default: self.max_hops)
        
        Returns:
            ReasoningPath with steps and answer
        """
        if max_hops is None:
            max_hops = self.max_hops
        
        # Parse query
        query_type, entities = self._parse_query(query)
        
        if not entities:
            return ReasoningPath(query=query, answer="Could not identify entities in query.")
        
        path = ReasoningPath(query=query)
        
        # Different strategies for different query types
        if query_type == 'WHY':
            path = self._reason_why(entities, query, max_hops)
        elif query_type == 'HOW':
            path = self._reason_how(entities, query, max_hops)
        elif query_type == 'RELATION':
            path = self._reason_relation(entities, query, max_hops)
        else:
            # Default: find connections
            path = self._reason_connections(entities[0], query, max_hops)
        
        return path
    
    def _parse_query(self, query: str) -> Tuple[str, List[str]]:
        """Parse query to determine type and extract entities."""
        query_lower = query.lower()
        
        # Determine query type
        if query_lower.startswith('why'):
            query_type = 'WHY'
        elif query_lower.startswith('how'):
            query_type = 'HOW'
        elif 'relationship' in query_lower or 'between' in query_lower:
            query_type = 'RELATION'
        else:
            query_type = 'GENERAL'
        
        # Main character names to prioritize
        main_characters = {
            'holmes', 'watson', 'moriarty', 'lestrade', 'mycroft', 'irene',
            'darcy', 'elizabeth', 'jane', 'bingley', 'wickham', 'lydia', 'collins',
            'charlotte', 'bennet', 'catherine', 'georgiana',
        }
        
        # Extract entities - prioritize main characters
        entities = []
        
        # First pass: main characters
        for entity in main_characters:
            if entity in query_lower:
                entities.append(entity)
        
        # Second pass: other known entities (if needed)
        if len(entities) < 2 and query_type == 'RELATION':
            for entity in self.entity_graph.keys():
                if entity in query_lower and len(entity) > 3 and entity not in entities:
                    entities.append(entity)
                    if len(entities) >= 2:
                        break
        
        return query_type, entities
    
    def _reason_why(self, entities: List[str], query: str, max_hops: int) -> ReasoningPath:
        """Reason about WHY questions - find causal chains."""
        path = ReasoningPath(query=query)
        
        if not entities:
            path.answer = "No entities found for WHY reasoning."
            return path
        
        start = entities[0]
        visited = {start}
        current = start
        
        # Look for causal indicators
        causal_actions = {'THINK', 'PERCEIVE', 'FEEL', 'ACT'}
        
        for hop in range(max_hops):
            neighbors = self.entity_graph.get(current, [])
            
            # Prefer causal actions
            best = None
            best_score = 0
            
            for neighbor, action, text in neighbors:
                if neighbor in visited:
                    continue
                
                score = 1.0
                if action in causal_actions:
                    score += 1.0
                if 'because' in text.lower() or 'reason' in text.lower():
                    score += 2.0
                
                if score > best_score:
                    best_score = score
                    best = (neighbor, action, text)
            
            if best is None:
                break
            
            neighbor, action, text = best
            step = ReasoningStep(
                entity=current,
                action=action,
                relation=neighbor,
                evidence=text,
                confidence=best_score / 4.0
            )
            path.add_step(step)
            
            visited.add(neighbor)
            current = neighbor
        
        # Generate answer
        if path.steps:
            chain = path.get_chain_description()
            path.answer = f"Based on the reasoning chain: {chain}"
        else:
            path.answer = f"Could not find causal chain for {start}."
        
        return path
    
    def _reason_how(self, entities: List[str], query: str, max_hops: int) -> ReasoningPath:
        """Reason about HOW questions - find process chains."""
        path = ReasoningPath(query=query)
        
        if not entities:
            path.answer = "No entities found for HOW reasoning."
            return path
        
        start = entities[0]
        visited = {start}
        current = start
        
        # Look for action/process indicators
        process_actions = {'ACT', 'MOVE', 'SPEAK'}
        
        for hop in range(max_hops):
            neighbors = self.entity_graph.get(current, [])
            
            best = None
            best_score = 0
            
            for neighbor, action, text in neighbors:
                if neighbor in visited:
                    continue
                
                score = 1.0
                if action in process_actions:
                    score += 1.0
                
                if score > best_score:
                    best_score = score
                    best = (neighbor, action, text)
            
            if best is None:
                break
            
            neighbor, action, text = best
            step = ReasoningStep(
                entity=current,
                action=action,
                relation=neighbor,
                evidence=text,
                confidence=best_score / 3.0
            )
            path.add_step(step)
            
            visited.add(neighbor)
            current = neighbor
        
        if path.steps:
            chain = path.get_chain_description()
            path.answer = f"The process involves: {chain}"
        else:
            path.answer = f"Could not find process chain for {start}."
        
        return path
    
    def _reason_relation(self, entities: List[str], query: str, max_hops: int) -> ReasoningPath:
        """Find the relationship path between two entities."""
        path = ReasoningPath(query=query)
        
        if len(entities) < 2:
            path.answer = "Need two entities to find relationship."
            return path
        
        start, end = entities[0], entities[1]
        
        # BFS to find shortest path
        from collections import deque
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue and len(visited) < 100:
            current, current_path = queue.popleft()
            
            if current == end:
                # Found path - build reasoning steps
                for i in range(len(current_path) - 1):
                    from_entity = current_path[i]
                    to_entity = current_path[i + 1]
                    
                    # Find the action between them
                    action = "RELATE"
                    evidence = ""
                    for neighbor, act, text in self.entity_graph.get(from_entity, []):
                        if neighbor == to_entity:
                            action = act
                            evidence = text
                            break
                    
                    step = ReasoningStep(
                        entity=from_entity,
                        action=action,
                        relation=to_entity,
                        evidence=evidence
                    )
                    path.add_step(step)
                
                path.answer = f"{start.title()} is connected to {end.title()} through {len(path)} steps."
                return path
            
            for neighbor, action, text in self.entity_graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_path + [neighbor]))
        
        path.answer = f"No path found between {start} and {end}."
        return path
    
    def _reason_connections(self, entity: str, query: str, max_hops: int) -> ReasoningPath:
        """Find interesting connections from an entity."""
        path = ReasoningPath(query=query)
        
        visited = {entity}
        current = entity
        
        for hop in range(max_hops):
            neighbors = self.entity_graph.get(current, [])
            
            if not neighbors:
                break
            
            # Pick most interesting neighbor (by action diversity)
            action_counts = Counter(action for _, action, _ in neighbors)
            best = None
            best_score = 0
            
            for neighbor, action, text in neighbors:
                if neighbor in visited:
                    continue
                
                # Prefer rare actions
                score = 1.0 / (action_counts[action] + 1)
                
                if score > best_score:
                    best_score = score
                    best = (neighbor, action, text)
            
            if best is None:
                break
            
            neighbor, action, text = best
            step = ReasoningStep(
                entity=current,
                action=action,
                relation=neighbor,
                evidence=text
            )
            path.add_step(step)
            
            visited.add(neighbor)
            current = neighbor
        
        if path.steps:
            entities = path.get_entities()
            path.answer = f"Connections from {entity}: {' → '.join(entities)}"
        else:
            path.answer = f"No connections found from {entity}."
        
        return path
    
    def get_entity_neighbors(self, entity: str, k: int = 5) -> List[Tuple[str, str]]:
        """Get k nearest neighbors of an entity."""
        neighbors = self.entity_graph.get(entity.lower(), [])
        
        # Count and sort
        counts = Counter()
        for neighbor, action, _ in neighbors:
            counts[(neighbor, action)] += 1
        
        return counts.most_common(k)


# =============================================================================
# TEST
# =============================================================================

def test_reasoning_engine():
    """Test the reasoning engine."""
    print("=== ReasoningEngine Test ===\n")
    
    from ..core.concept_knowledge import ConceptKnowledge
    
    # Create mock knowledge
    kb = ConceptKnowledge()
    
    # Add some test frames
    test_frames = [
        {'agent': 'holmes', 'action': 'PERCEIVE', 'patient': 'clue', 'text': 'Holmes noticed the clue'},
        {'agent': 'holmes', 'action': 'THINK', 'patient': 'butler', 'text': 'Holmes suspected the butler'},
        {'agent': 'butler', 'action': 'ACT', 'patient': 'crime', 'text': 'The butler committed the crime'},
        {'agent': 'watson', 'action': 'SPEAK', 'patient': 'holmes', 'text': 'Watson spoke to Holmes'},
        {'agent': 'holmes', 'action': 'SPEAK', 'patient': 'watson', 'text': 'Holmes replied to Watson'},
    ]
    
    kb.frames = test_frames
    
    # Create engine
    engine = ReasoningEngine(kb)
    
    print("Entity graph:")
    for entity, neighbors in engine.entity_graph.items():
        print(f"  {entity}: {len(neighbors)} connections")
    print()
    
    # Test reasoning
    queries = [
        "Why did Holmes suspect the butler?",
        "How did Holmes solve the case?",
        "What is the relationship between Holmes and Watson?",
    ]
    
    for query in queries:
        print(f"Q: {query}")
        path = engine.reason(query)
        print(f"A: {path.answer}")
        if path.steps:
            print(f"   Chain: {path.get_chain_description()}")
        print()


if __name__ == '__main__':
    test_reasoning_engine()
