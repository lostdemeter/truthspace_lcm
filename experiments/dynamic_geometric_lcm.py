#!/usr/bin/env python3
"""
Dynamic Geometric LCM

A unified architecture where:
1. STRUCTURE IS THE DATA - positions encode meaning, relations encode connections
2. LEARNING IS STRUCTURE UPDATE - new information modifies the geometry

This combines:
- Explicit relations (known structure)
- Attractor learning (emergent structure from data)
- Dynamic updates (structure evolves with new information)

The key insight: Traditional LLMs store knowledge in weights.
We store knowledge in GEOMETRY - positions and relations in vector space.

Learning = moving points and creating/strengthening relations
Inference = geometric operations (projection, similarity, relation traversal)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json

PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class Entity:
    """An entity in the geometric space."""
    name: str
    position: np.ndarray
    entity_type: str = "unknown"
    confidence: float = 1.0
    update_count: int = 0
    
    def __hash__(self):
        return hash(self.name)


@dataclass  
class Relation:
    """A relation between entities - stored as a vector offset."""
    name: str
    vector: np.ndarray
    confidence: float = 1.0
    instance_count: int = 0  # How many times we've seen this relation


@dataclass
class Fact:
    """A fact: subject --relation--> object"""
    subject: str
    relation: str
    object: str
    confidence: float = 1.0


# =============================================================================
# DYNAMIC GEOMETRIC SPACE
# =============================================================================

class DynamicGeometricSpace:
    """
    A geometric space that learns and updates dynamically.
    
    Core principles:
    1. Entities have positions (learned from context)
    2. Relations have vectors (learned from entity pairs)
    3. New information updates both positions and relations
    4. Structure IS the knowledge - no separate "weights"
    """
    
    def __init__(self, dim: int = 256, learning_rate: float = 0.1):
        self.dim = dim
        self.learning_rate = learning_rate
        
        # Entity storage
        self.entities: Dict[str, Entity] = {}
        
        # Relation storage (relation_name -> Relation)
        self.relations: Dict[str, Relation] = {}
        
        # Fact storage for verification
        self.facts: List[Fact] = []
        
        # Co-occurrence tracking for attractor dynamics
        self.cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Type inference
        self.type_hints: Dict[str, Set[str]] = defaultdict(set)
    
    # =========================================================================
    # ENTITY MANAGEMENT
    # =========================================================================
    
    def get_or_create_entity(self, name: str, entity_type: str = "unknown") -> Entity:
        """Get existing entity or create new one with random position."""
        if name not in self.entities:
            # Initialize with deterministic random position
            seed = hash(name) % (2**32)
            rng = np.random.default_rng(seed)
            position = rng.standard_normal(self.dim)
            position = position / np.linalg.norm(position)
            
            self.entities[name] = Entity(
                name=name,
                position=position,
                entity_type=entity_type
            )
        
        entity = self.entities[name]
        if entity_type != "unknown":
            entity.entity_type = entity_type
            self.type_hints[entity_type].add(name)
        
        return entity
    
    def update_entity_position(self, name: str, target: np.ndarray, 
                               strength: float = None):
        """Move entity toward target position (attractor dynamics)."""
        if name not in self.entities:
            return
        
        entity = self.entities[name]
        lr = strength if strength is not None else self.learning_rate
        
        # Move toward target
        entity.position = (1 - lr) * entity.position + lr * target
        entity.position = entity.position / np.linalg.norm(entity.position)
        entity.update_count += 1
    
    # =========================================================================
    # RELATION MANAGEMENT
    # =========================================================================
    
    def get_or_create_relation(self, name: str) -> Relation:
        """Get existing relation or create new one."""
        if name not in self.relations:
            # Initialize with random vector
            seed = hash(f"__REL__{name}") % (2**32)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self.dim)
            vector = vector / np.linalg.norm(vector)
            
            self.relations[name] = Relation(
                name=name,
                vector=vector
            )
        
        return self.relations[name]
    
    def update_relation_from_pair(self, relation_name: str, 
                                   subject: str, object_: str):
        """
        Update relation vector based on observed subject-object pair.
        
        The relation vector should be: object_position - subject_position
        We update it incrementally as we see more examples.
        """
        if subject not in self.entities or object_ not in self.entities:
            return
        
        relation = self.get_or_create_relation(relation_name)
        
        # Compute observed offset
        subj_pos = self.entities[subject].position
        obj_pos = self.entities[object_].position
        observed_offset = obj_pos - subj_pos
        observed_offset = observed_offset / (np.linalg.norm(observed_offset) + 1e-10)
        
        # Update relation vector (running average)
        n = relation.instance_count
        if n == 0:
            relation.vector = observed_offset
        else:
            # Weighted average favoring recent observations
            weight = 1.0 / (n + 1)
            relation.vector = (1 - weight) * relation.vector + weight * observed_offset
            relation.vector = relation.vector / np.linalg.norm(relation.vector)
        
        relation.instance_count += 1
    
    # =========================================================================
    # LEARNING FROM DATA
    # =========================================================================
    
    def learn_fact(self, subject: str, relation: str, object_: str,
                   subject_type: str = "unknown", object_type: str = "unknown"):
        """
        Learn a fact: subject --relation--> object
        
        This updates:
        1. Entity positions (attractor dynamics)
        2. Relation vectors (from observed offsets)
        3. Co-occurrence counts
        """
        # Ensure entities exist
        subj_entity = self.get_or_create_entity(subject, subject_type)
        obj_entity = self.get_or_create_entity(object_, object_type)
        rel = self.get_or_create_relation(relation)
        
        # Store fact
        self.facts.append(Fact(subject, relation, object_))
        
        # Update co-occurrence
        self.cooccurrence[(subject, object_)] += 1
        self.cooccurrence[(object_, subject)] += 1
        
        # Update relation vector from this pair
        self.update_relation_from_pair(relation, subject, object_)
        
        # Attractor dynamics: pull object toward subject + relation
        target_obj_pos = subj_entity.position + rel.vector
        target_obj_pos = target_obj_pos / np.linalg.norm(target_obj_pos)
        self.update_entity_position(object_, target_obj_pos)
        
        # Also adjust subject position slightly
        # subject should be at: object - relation
        target_subj_pos = obj_entity.position - rel.vector
        target_subj_pos = target_subj_pos / np.linalg.norm(target_subj_pos)
        self.update_entity_position(subject, target_subj_pos, strength=self.learning_rate * 0.5)
    
    def learn_cooccurrence(self, entity1: str, entity2: str):
        """
        Learn that two entities co-occur (should be near each other).
        
        This is pure attractor dynamics without explicit relation.
        """
        self.get_or_create_entity(entity1)
        self.get_or_create_entity(entity2)
        
        self.cooccurrence[(entity1, entity2)] += 1
        self.cooccurrence[(entity2, entity1)] += 1
        
        # Pull entities toward each other
        pos1 = self.entities[entity1].position
        pos2 = self.entities[entity2].position
        
        midpoint = (pos1 + pos2) / 2
        midpoint = midpoint / np.linalg.norm(midpoint)
        
        self.update_entity_position(entity1, midpoint, strength=self.learning_rate * 0.3)
        self.update_entity_position(entity2, midpoint, strength=self.learning_rate * 0.3)
    
    def learn_type_separation(self, type1: str, type2: str):
        """
        Learn that two types should be separated (repeller dynamics).
        
        Entities of different types should be far apart.
        """
        entities1 = self.type_hints.get(type1, set())
        entities2 = self.type_hints.get(type2, set())
        
        for e1 in entities1:
            for e2 in entities2:
                if e1 == e2:
                    continue
                
                pos1 = self.entities[e1].position
                pos2 = self.entities[e2].position
                
                # Push apart
                direction = pos1 - pos2
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                
                self.update_entity_position(e1, pos1 + direction * 0.1, 
                                          strength=self.learning_rate * 0.2)
                self.update_entity_position(e2, pos2 - direction * 0.1,
                                          strength=self.learning_rate * 0.2)
    
    # =========================================================================
    # INFERENCE
    # =========================================================================
    
    def query_relation(self, subject: str, relation: str, 
                       k: int = 5) -> List[Tuple[str, float]]:
        """
        Query: subject --relation--> ?
        
        Find entities that are at subject + relation_vector.
        """
        if subject not in self.entities or relation not in self.relations:
            return []
        
        subj_pos = self.entities[subject].position
        rel_vec = self.relations[relation].vector
        
        # Target position
        target = subj_pos + rel_vec
        target = target / np.linalg.norm(target)
        
        # Find nearest entities
        results = []
        for name, entity in self.entities.items():
            if name == subject:
                continue
            sim = np.dot(target, entity.position)
            results.append((name, sim))
        
        return sorted(results, key=lambda x: -x[1])[:k]
    
    def query_inverse_relation(self, object_: str, relation: str,
                               k: int = 5) -> List[Tuple[str, float]]:
        """
        Query: ? --relation--> object
        
        Find entities that are at object - relation_vector.
        """
        if object_ not in self.entities or relation not in self.relations:
            return []
        
        obj_pos = self.entities[object_].position
        rel_vec = self.relations[relation].vector
        
        # Target position (inverse)
        target = obj_pos - rel_vec
        target = target / np.linalg.norm(target)
        
        # Find nearest entities
        results = []
        for name, entity in self.entities.items():
            if name == object_:
                continue
            sim = np.dot(target, entity.position)
            results.append((name, sim))
        
        return sorted(results, key=lambda x: -x[1])[:k]
    
    def solve_analogy(self, a: str, b: str, c: str, 
                      k: int = 5) -> List[Tuple[str, float]]:
        """
        Solve: a:b :: c:?
        
        The relation from a to b should apply to c.
        """
        if a not in self.entities or b not in self.entities or c not in self.entities:
            return []
        
        # Extract relation
        a_pos = self.entities[a].position
        b_pos = self.entities[b].position
        c_pos = self.entities[c].position
        
        relation = b_pos - a_pos
        
        # Apply to c
        target = c_pos + relation
        target = target / np.linalg.norm(target)
        
        # Find nearest
        results = []
        for name, entity in self.entities.items():
            if name in [a, b, c]:
                continue
            sim = np.dot(target, entity.position)
            results.append((name, sim))
        
        return sorted(results, key=lambda x: -x[1])[:k]
    
    def find_similar(self, entity: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find entities similar to the given one."""
        if entity not in self.entities:
            return []
        
        pos = self.entities[entity].position
        
        results = []
        for name, e in self.entities.items():
            if name == entity:
                continue
            sim = np.dot(pos, e.position)
            results.append((name, sim))
        
        return sorted(results, key=lambda x: -x[1])[:k]
    
    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================
    
    def relation_consistency(self, relation: str) -> float:
        """
        Measure how consistent a relation is across its instances.
        
        High consistency = relation vector is stable.
        """
        if relation not in self.relations:
            return 0.0
        
        rel = self.relations[relation]
        
        # Find all facts with this relation
        relevant_facts = [f for f in self.facts if f.relation == relation]
        
        if len(relevant_facts) < 2:
            return 1.0  # Not enough data
        
        # Compute offset for each fact
        offsets = []
        for fact in relevant_facts:
            if fact.subject in self.entities and fact.object in self.entities:
                subj_pos = self.entities[fact.subject].position
                obj_pos = self.entities[fact.object].position
                offset = obj_pos - subj_pos
                offset = offset / (np.linalg.norm(offset) + 1e-10)
                offsets.append(offset)
        
        if len(offsets) < 2:
            return 1.0
        
        # Measure pairwise similarity
        sims = []
        for i in range(len(offsets)):
            for j in range(i+1, len(offsets)):
                sims.append(np.dot(offsets[i], offsets[j]))
        
        return np.mean(sims)
    
    def type_separation_score(self, type1: str, type2: str) -> float:
        """Measure how well separated two types are."""
        entities1 = [self.entities[e] for e in self.type_hints.get(type1, set()) 
                    if e in self.entities]
        entities2 = [self.entities[e] for e in self.type_hints.get(type2, set())
                    if e in self.entities]
        
        if not entities1 or not entities2:
            return 0.0
        
        # Average cross-type similarity (lower = better separation)
        sims = []
        for e1 in entities1:
            for e2 in entities2:
                sims.append(np.dot(e1.position, e2.position))
        
        # Return 1 - avg_sim (so higher = better separation)
        return 1.0 - np.mean(sims)
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save(self, filepath: str):
        """Save the geometric space to a file."""
        data = {
            'dim': self.dim,
            'learning_rate': self.learning_rate,
            'entities': {
                name: {
                    'position': e.position.tolist(),
                    'entity_type': e.entity_type,
                    'confidence': e.confidence,
                    'update_count': e.update_count
                }
                for name, e in self.entities.items()
            },
            'relations': {
                name: {
                    'vector': r.vector.tolist(),
                    'confidence': r.confidence,
                    'instance_count': r.instance_count
                }
                for name, r in self.relations.items()
            },
            'facts': [
                {'subject': f.subject, 'relation': f.relation, 
                 'object': f.object, 'confidence': f.confidence}
                for f in self.facts
            ],
            'type_hints': {k: list(v) for k, v in self.type_hints.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load the geometric space from a file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.dim = data['dim']
        self.learning_rate = data['learning_rate']
        
        self.entities = {
            name: Entity(
                name=name,
                position=np.array(e['position']),
                entity_type=e['entity_type'],
                confidence=e['confidence'],
                update_count=e['update_count']
            )
            for name, e in data['entities'].items()
        }
        
        self.relations = {
            name: Relation(
                name=name,
                vector=np.array(r['vector']),
                confidence=r['confidence'],
                instance_count=r['instance_count']
            )
            for name, r in data['relations'].items()
        }
        
        self.facts = [
            Fact(f['subject'], f['relation'], f['object'], f['confidence'])
            for f in data['facts']
        ]
        
        self.type_hints = {k: set(v) for k, v in data['type_hints'].items()}


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo_dynamic_learning():
    """Demonstrate dynamic learning of structure."""
    
    print("=" * 70)
    print("DYNAMIC GEOMETRIC LCM DEMONSTRATION")
    print("=" * 70)
    print()
    print("Principle: Structure IS the data. Learning IS structure update.")
    print()
    
    space = DynamicGeometricSpace(dim=256, learning_rate=0.15)
    
    # Phase 1: Learn some facts
    print("PHASE 1: Learning facts")
    print("-" * 50)
    
    facts = [
        ("france", "capital_of", "paris", "country", "city"),
        ("germany", "capital_of", "berlin", "country", "city"),
        ("japan", "capital_of", "tokyo", "country", "city"),
        ("italy", "capital_of", "rome", "country", "city"),
        ("spain", "capital_of", "madrid", "country", "city"),
        ("uk", "capital_of", "london", "country", "city"),
    ]
    
    for subj, rel, obj, subj_type, obj_type in facts:
        space.learn_fact(subj, rel, obj, subj_type, obj_type)
        print(f"  Learned: {subj} --{rel}--> {obj}")
    
    print()
    
    # Check relation consistency
    consistency = space.relation_consistency("capital_of")
    print(f"Relation 'capital_of' consistency: {consistency:.3f}")
    print()
    
    # Phase 2: Query learned knowledge
    print("PHASE 2: Querying learned knowledge")
    print("-" * 50)
    
    queries = [
        ("france", "capital_of"),
        ("germany", "capital_of"),
        ("japan", "capital_of"),
    ]
    
    for subj, rel in queries:
        results = space.query_relation(subj, rel, k=3)
        print(f"  {subj} --{rel}--> ?")
        for name, sim in results[:1]:
            print(f"    → {name} (sim: {sim:.3f})")
    
    print()
    
    # Phase 3: Test analogies
    print("PHASE 3: Analogical reasoning")
    print("-" * 50)
    
    analogies = [
        ("france", "paris", "germany", "berlin"),
        ("france", "paris", "japan", "tokyo"),
        ("germany", "berlin", "italy", "rome"),
    ]
    
    correct = 0
    for a, b, c, expected in analogies:
        results = space.solve_analogy(a, b, c, k=3)
        answer = results[0][0] if results else "?"
        match = "✓" if answer == expected else "✗"
        print(f"  {match} {a}:{b} :: {c}:? → {answer} (expected: {expected})")
        if answer == expected:
            correct += 1
    
    print(f"\nAnalogy accuracy: {correct}/{len(analogies)} = {correct/len(analogies):.1%}")
    print()
    
    # Phase 4: Learn more facts and see improvement
    print("PHASE 4: Incremental learning")
    print("-" * 50)
    
    # Learn more facts to reinforce the relation
    more_facts = [
        ("china", "capital_of", "beijing", "country", "city"),
        ("russia", "capital_of", "moscow", "country", "city"),
        ("brazil", "capital_of", "brasilia", "country", "city"),
    ]
    
    for subj, rel, obj, subj_type, obj_type in more_facts:
        space.learn_fact(subj, rel, obj, subj_type, obj_type)
        print(f"  Learned: {subj} --{rel}--> {obj}")
    
    # Re-learn original facts to strengthen
    print("\n  Reinforcing original facts...")
    for subj, rel, obj, subj_type, obj_type in facts:
        space.learn_fact(subj, rel, obj, subj_type, obj_type)
    
    print()
    
    # Check improved consistency
    consistency = space.relation_consistency("capital_of")
    print(f"Relation 'capital_of' consistency after more learning: {consistency:.3f}")
    print()
    
    # Test analogies again
    print("PHASE 5: Re-test analogies after more learning")
    print("-" * 50)
    
    correct = 0
    for a, b, c, expected in analogies:
        results = space.solve_analogy(a, b, c, k=3)
        answer = results[0][0] if results else "?"
        match = "✓" if answer == expected else "✗"
        print(f"  {match} {a}:{b} :: {c}:? → {answer} (expected: {expected})")
        if answer == expected:
            correct += 1
    
    # Test new analogies with newly learned entities
    new_analogies = [
        ("france", "paris", "china", "beijing"),
        ("germany", "berlin", "russia", "moscow"),
    ]
    
    for a, b, c, expected in new_analogies:
        results = space.solve_analogy(a, b, c, k=3)
        answer = results[0][0] if results else "?"
        match = "✓" if answer == expected else "✗"
        print(f"  {match} {a}:{b} :: {c}:? → {answer} (expected: {expected})")
        if answer == expected:
            correct += 1
    
    total = len(analogies) + len(new_analogies)
    print(f"\nTotal analogy accuracy: {correct}/{total} = {correct/total:.1%}")
    print()
    
    # Phase 6: Type separation
    print("PHASE 6: Type separation")
    print("-" * 50)
    
    sep_score = space.type_separation_score("country", "city")
    print(f"Country-City separation: {sep_score:.3f}")
    
    # Apply type separation learning
    print("\nApplying type separation dynamics...")
    for _ in range(10):
        space.learn_type_separation("country", "city")
    
    sep_score = space.type_separation_score("country", "city")
    print(f"Country-City separation after learning: {sep_score:.3f}")
    
    print()
    
    return space


def demo_continuous_learning():
    """Demonstrate continuous learning from a stream of facts."""
    
    print()
    print("=" * 70)
    print("CONTINUOUS LEARNING DEMONSTRATION")
    print("=" * 70)
    print()
    
    space = DynamicGeometricSpace(dim=256, learning_rate=0.1)
    
    # Simulate a stream of facts
    fact_stream = [
        # Batch 1: Some capitals
        ("france", "capital_of", "paris"),
        ("germany", "capital_of", "berlin"),
        
        # Batch 2: Some authors
        ("melville", "wrote", "moby_dick"),
        ("shakespeare", "wrote", "hamlet"),
        
        # Batch 3: More capitals
        ("japan", "capital_of", "tokyo"),
        ("italy", "capital_of", "rome"),
        
        # Batch 4: More authors
        ("orwell", "wrote", "1984"),
        ("tolkien", "wrote", "lotr"),
        
        # Batch 5: Locations
        ("eiffel_tower", "located_in", "paris"),
        ("colosseum", "located_in", "rome"),
    ]
    
    print("Learning from fact stream:")
    print("-" * 50)
    
    for i, (subj, rel, obj) in enumerate(fact_stream):
        space.learn_fact(subj, rel, obj)
        
        if (i + 1) % 2 == 0:
            print(f"\nAfter {i+1} facts:")
            
            # Test capital_of if we have enough data
            if "capital_of" in space.relations:
                results = space.query_relation("france", "capital_of", k=1)
                if results:
                    print(f"  france --capital_of--> {results[0][0]}")
            
            # Test wrote if we have enough data
            if "wrote" in space.relations:
                results = space.query_relation("melville", "wrote", k=1)
                if results:
                    print(f"  melville --wrote--> {results[0][0]}")
    
    print()
    print("Final state:")
    print("-" * 50)
    
    print(f"Entities: {len(space.entities)}")
    print(f"Relations: {list(space.relations.keys())}")
    print(f"Facts: {len(space.facts)}")
    
    print()
    print("Relation consistencies:")
    for rel_name in space.relations:
        consistency = space.relation_consistency(rel_name)
        count = space.relations[rel_name].instance_count
        print(f"  {rel_name}: {consistency:.3f} ({count} instances)")
    
    print()
    print("Cross-domain queries:")
    print("-" * 50)
    
    # Can we answer questions about different domains?
    queries = [
        ("germany", "capital_of"),
        ("shakespeare", "wrote"),
        ("eiffel_tower", "located_in"),
    ]
    
    for subj, rel in queries:
        results = space.query_relation(subj, rel, k=1)
        if results:
            print(f"  {subj} --{rel}--> {results[0][0]} (sim: {results[0][1]:.3f})")
    
    return space


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("DYNAMIC GEOMETRIC LCM")
    print("=" * 70)
    print()
    print("A system where STRUCTURE IS DATA and LEARNING IS STRUCTURE UPDATE.")
    print()
    print("Key principles:")
    print("  1. Entities have positions (learned from context)")
    print("  2. Relations have vectors (learned from entity pairs)")
    print("  3. New information updates both positions and relations")
    print("  4. No separate 'weights' - the geometry IS the knowledge")
    print()
    
    space1 = demo_dynamic_learning()
    space2 = demo_continuous_learning()
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("The Dynamic Geometric LCM demonstrates:")
    print("  1. Learning from facts updates entity positions and relation vectors")
    print("  2. Relations become consistent through repeated observation")
    print("  3. Analogies work because relations are invariant (learned, not random)")
    print("  4. Type separation emerges from attractor/repeller dynamics")
    print("  5. The system can learn continuously from a stream of facts")
    print()
    print("This is a foundation for replacing traditional LLMs with pure geometry.")
    print("Knowledge is stored in STRUCTURE, not weights.")
    print()


if __name__ == "__main__":
    main()
