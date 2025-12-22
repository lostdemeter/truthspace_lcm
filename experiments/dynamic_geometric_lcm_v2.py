#!/usr/bin/env python3
"""
Dynamic Geometric LCM v2

Improvements over v1:
1. Better convergence through iterative refinement
2. Explicit relation anchoring (combine explicit + learned)
3. Batch learning with multiple passes
4. Consistency-driven learning rate

Key insight from v1: Single-pass learning gives ~0.2 consistency.
We need iterative refinement to reach high consistency.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2


@dataclass
class Entity:
    """An entity in the geometric space."""
    name: str
    position: np.ndarray
    entity_type: str = "unknown"
    frozen: bool = False  # If True, position doesn't update


@dataclass  
class Relation:
    """A relation - combines explicit definition with learned refinement."""
    name: str
    vector: np.ndarray
    is_explicit: bool = False  # True if explicitly defined
    instance_count: int = 0
    consistency: float = 0.0


class DynamicGeometricLCM:
    """
    Dynamic Geometric LCM with improved learning.
    
    Key improvements:
    1. Iterative batch learning for convergence
    2. Explicit relation anchoring
    3. Consistency-aware updates
    """
    
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.facts: List[Tuple[str, str, str]] = []  # (subj, rel, obj)
        self.type_hints: Dict[str, Set[str]] = defaultdict(set)
    
    # =========================================================================
    # ENTITY MANAGEMENT
    # =========================================================================
    
    def get_entity(self, name: str, create: bool = True) -> Optional[Entity]:
        """Get or create entity."""
        if name in self.entities:
            return self.entities[name]
        
        if not create:
            return None
        
        # Deterministic initialization
        seed = hash(name) % (2**32)
        rng = np.random.default_rng(seed)
        pos = rng.standard_normal(self.dim)
        pos = pos / np.linalg.norm(pos)
        
        self.entities[name] = Entity(name=name, position=pos)
        return self.entities[name]
    
    def set_entity_type(self, name: str, entity_type: str):
        """Set entity type for type-based operations."""
        entity = self.get_entity(name)
        entity.entity_type = entity_type
        self.type_hints[entity_type].add(name)
    
    # =========================================================================
    # RELATION MANAGEMENT
    # =========================================================================
    
    def define_relation(self, name: str, vector: np.ndarray = None):
        """
        Explicitly define a relation.
        
        If vector is None, creates a random one that will be refined.
        """
        if vector is None:
            seed = hash(f"__REL__{name}") % (2**32)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self.dim)
            vector = vector / np.linalg.norm(vector)
        
        self.relations[name] = Relation(
            name=name,
            vector=vector,
            is_explicit=True
        )
    
    def get_relation(self, name: str) -> Relation:
        """Get or create relation."""
        if name not in self.relations:
            self.define_relation(name)
        return self.relations[name]
    
    # =========================================================================
    # FACT MANAGEMENT
    # =========================================================================
    
    def add_fact(self, subject: str, relation: str, object_: str,
                 subject_type: str = None, object_type: str = None):
        """Add a fact to be learned."""
        self.get_entity(subject)
        self.get_entity(object_)
        self.get_relation(relation)
        
        self.facts.append((subject, relation, object_))
        
        if subject_type:
            self.set_entity_type(subject, subject_type)
        if object_type:
            self.set_entity_type(object_, object_type)
    
    # =========================================================================
    # LEARNING
    # =========================================================================
    
    def learn(self, n_iterations: int = 100, target_consistency: float = 0.95,
              verbose: bool = False):
        """
        Learn entity positions and relation vectors from facts.
        
        Iterates until consistency reaches target or max iterations.
        """
        if not self.facts:
            return
        
        # Group facts by relation
        facts_by_relation = defaultdict(list)
        for subj, rel, obj in self.facts:
            facts_by_relation[rel].append((subj, obj))
        
        for iteration in range(n_iterations):
            # Adaptive learning rate (decreases as we converge)
            lr = 0.3 * (1.0 - iteration / n_iterations)
            
            # Step 1: Update relation vectors from current positions
            for rel_name, pairs in facts_by_relation.items():
                self._update_relation_vector(rel_name, pairs)
            
            # Step 2: Update entity positions to align with relations
            for subj, rel, obj in self.facts:
                self._update_positions_for_fact(subj, rel, obj, lr)
            
            # Step 3: Check consistency
            min_consistency = 1.0
            for rel_name in facts_by_relation:
                consistency = self._compute_consistency(rel_name)
                self.relations[rel_name].consistency = consistency
                min_consistency = min(min_consistency, consistency)
            
            if verbose and iteration % 10 == 0:
                print(f"  Iteration {iteration}: min consistency = {min_consistency:.3f}")
            
            if min_consistency >= target_consistency:
                if verbose:
                    print(f"  Converged at iteration {iteration}")
                break
        
        return min_consistency
    
    def _update_relation_vector(self, rel_name: str, pairs: List[Tuple[str, str]]):
        """Update relation vector as average of observed offsets."""
        offsets = []
        for subj, obj in pairs:
            subj_pos = self.entities[subj].position
            obj_pos = self.entities[obj].position
            offset = obj_pos - subj_pos
            norm = np.linalg.norm(offset)
            if norm > 1e-10:
                offsets.append(offset / norm)
        
        if offsets:
            avg_offset = np.mean(offsets, axis=0)
            avg_offset = avg_offset / np.linalg.norm(avg_offset)
            
            rel = self.relations[rel_name]
            # Blend with existing (for stability)
            rel.vector = 0.7 * avg_offset + 0.3 * rel.vector
            rel.vector = rel.vector / np.linalg.norm(rel.vector)
            rel.instance_count = len(pairs)
    
    def _update_positions_for_fact(self, subj: str, rel: str, obj: str, lr: float):
        """Update entity positions to align with relation."""
        subj_entity = self.entities[subj]
        obj_entity = self.entities[obj]
        rel_vec = self.relations[rel].vector
        
        if not subj_entity.frozen:
            # Subject should be at: object - relation
            target_subj = obj_entity.position - rel_vec
            target_subj = target_subj / np.linalg.norm(target_subj)
            subj_entity.position = (1 - lr*0.5) * subj_entity.position + lr*0.5 * target_subj
            subj_entity.position = subj_entity.position / np.linalg.norm(subj_entity.position)
        
        if not obj_entity.frozen:
            # Object should be at: subject + relation
            target_obj = subj_entity.position + rel_vec
            target_obj = target_obj / np.linalg.norm(target_obj)
            obj_entity.position = (1 - lr) * obj_entity.position + lr * target_obj
            obj_entity.position = obj_entity.position / np.linalg.norm(obj_entity.position)
    
    def _compute_consistency(self, rel_name: str) -> float:
        """Compute consistency of a relation across its instances."""
        pairs = [(s, o) for s, r, o in self.facts if r == rel_name]
        
        if len(pairs) < 2:
            return 1.0
        
        offsets = []
        for subj, obj in pairs:
            offset = self.entities[obj].position - self.entities[subj].position
            norm = np.linalg.norm(offset)
            if norm > 1e-10:
                offsets.append(offset / norm)
        
        if len(offsets) < 2:
            return 1.0
        
        # Pairwise similarity
        sims = []
        for i in range(len(offsets)):
            for j in range(i+1, len(offsets)):
                sims.append(np.dot(offsets[i], offsets[j]))
        
        return np.mean(sims)
    
    # =========================================================================
    # INFERENCE
    # =========================================================================
    
    def query(self, subject: str, relation: str, k: int = 5) -> List[Tuple[str, float]]:
        """Query: subject --relation--> ?"""
        if subject not in self.entities or relation not in self.relations:
            return []
        
        target = self.entities[subject].position + self.relations[relation].vector
        target = target / np.linalg.norm(target)
        
        results = []
        for name, entity in self.entities.items():
            if name == subject:
                continue
            sim = np.dot(target, entity.position)
            results.append((name, sim))
        
        return sorted(results, key=lambda x: -x[1])[:k]
    
    def analogy(self, a: str, b: str, c: str, k: int = 5) -> List[Tuple[str, float]]:
        """Solve: a:b :: c:?"""
        if a not in self.entities or b not in self.entities or c not in self.entities:
            return []
        
        relation = self.entities[b].position - self.entities[a].position
        target = self.entities[c].position + relation
        target = target / np.linalg.norm(target)
        
        results = []
        for name, entity in self.entities.items():
            if name in [a, b, c]:
                continue
            sim = np.dot(target, entity.position)
            results.append((name, sim))
        
        return sorted(results, key=lambda x: -x[1])[:k]
    
    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================
    
    def print_status(self):
        """Print current state."""
        print(f"Entities: {len(self.entities)}")
        print(f"Relations: {len(self.relations)}")
        print(f"Facts: {len(self.facts)}")
        print()
        print("Relation consistencies:")
        for name, rel in self.relations.items():
            print(f"  {name}: {rel.consistency:.3f} ({rel.instance_count} instances)")


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate the improved dynamic learning."""
    
    print("=" * 70)
    print("DYNAMIC GEOMETRIC LCM v2")
    print("=" * 70)
    print()
    
    lcm = DynamicGeometricLCM(dim=256)
    
    # Add facts
    print("Adding facts...")
    
    capitals = [
        ("france", "paris"),
        ("germany", "berlin"),
        ("japan", "tokyo"),
        ("italy", "rome"),
        ("spain", "madrid"),
        ("uk", "london"),
        ("china", "beijing"),
        ("russia", "moscow"),
    ]
    
    for country, capital in capitals:
        lcm.add_fact(country, "capital_of", capital, "country", "city")
    
    authors = [
        ("melville", "moby_dick"),
        ("shakespeare", "hamlet"),
        ("orwell", "1984"),
        ("tolkien", "lotr"),
    ]
    
    for author, book in authors:
        lcm.add_fact(author, "wrote", book, "author", "book")
    
    print(f"Added {len(lcm.facts)} facts")
    print()
    
    # Learn
    print("Learning (iterative refinement)...")
    print("-" * 50)
    
    final_consistency = lcm.learn(n_iterations=100, target_consistency=0.95, verbose=True)
    
    print()
    lcm.print_status()
    print()
    
    # Test queries
    print("Testing queries:")
    print("-" * 50)
    
    queries = [
        ("france", "capital_of"),
        ("germany", "capital_of"),
        ("melville", "wrote"),
        ("shakespeare", "wrote"),
    ]
    
    for subj, rel in queries:
        results = lcm.query(subj, rel, k=1)
        if results:
            print(f"  {subj} --{rel}--> {results[0][0]} (sim: {results[0][1]:.3f})")
    
    print()
    
    # Test analogies
    print("Testing analogies:")
    print("-" * 50)
    
    analogies = [
        ("france", "paris", "germany", "berlin"),
        ("france", "paris", "japan", "tokyo"),
        ("france", "paris", "china", "beijing"),
        ("germany", "berlin", "russia", "moscow"),
        ("melville", "moby_dick", "shakespeare", "hamlet"),
        ("melville", "moby_dick", "orwell", "1984"),
    ]
    
    correct = 0
    for a, b, c, expected in analogies:
        results = lcm.analogy(a, b, c, k=1)
        answer = results[0][0] if results else "?"
        match = "✓" if answer == expected else "✗"
        print(f"  {match} {a}:{b} :: {c}:? → {answer} (expected: {expected})")
        if answer == expected:
            correct += 1
    
    print()
    print(f"Analogy accuracy: {correct}/{len(analogies)} = {correct/len(analogies):.1%}")
    print()
    
    # Test incremental learning
    print("Testing incremental learning:")
    print("-" * 50)
    
    # Add new facts
    new_facts = [
        ("brazil", "capital_of", "brasilia", "country", "city"),
        ("india", "capital_of", "delhi", "country", "city"),
        ("hemingway", "wrote", "old_man_sea", "author", "book"),
    ]
    
    for subj, rel, obj, st, ot in new_facts:
        lcm.add_fact(subj, rel, obj, st, ot)
        print(f"  Added: {subj} --{rel}--> {obj}")
    
    print()
    print("Re-learning with new facts...")
    lcm.learn(n_iterations=50, verbose=True)
    
    print()
    
    # Test new analogies
    print("Testing analogies with new entities:")
    print("-" * 50)
    
    new_analogies = [
        ("france", "paris", "brazil", "brasilia"),
        ("france", "paris", "india", "delhi"),
        ("melville", "moby_dick", "hemingway", "old_man_sea"),
    ]
    
    correct = 0
    for a, b, c, expected in new_analogies:
        results = lcm.analogy(a, b, c, k=1)
        answer = results[0][0] if results else "?"
        match = "✓" if answer == expected else "✗"
        print(f"  {match} {a}:{b} :: {c}:? → {answer} (expected: {expected})")
        if answer == expected:
            correct += 1
    
    print()
    print(f"New analogy accuracy: {correct}/{len(new_analogies)} = {correct/len(new_analogies):.1%}")
    
    return lcm


if __name__ == "__main__":
    demo()
