#!/usr/bin/env python3
"""
Geometric LCM - Full Implementation

A complete geometric language model with:
1. Natural language parsing to facts
2. Multi-hop reasoning
3. Integration with TruthSpace vocabulary
4. Scaling support

This builds on dynamic_geometric_lcm_v2.py to create a more complete system.
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Entity:
    """An entity in the geometric space."""
    name: str
    position: np.ndarray
    entity_type: str = "unknown"
    aliases: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)


@dataclass
class Relation:
    """A relation between entities."""
    name: str
    vector: np.ndarray
    inverse_name: str = None  # e.g., "capital_of" inverse is "has_capital"
    consistency: float = 0.0
    instance_count: int = 0


@dataclass
class Fact:
    """A fact: subject --relation--> object"""
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    source: str = ""


# =============================================================================
# NATURAL LANGUAGE PARSER
# =============================================================================

class NLParser:
    """
    Parse natural language into facts.
    
    Patterns supported:
    - "X is the Y of Z" → (Z, Y, X)
    - "X is Y" → (X, is_a, Y)
    - "X wrote Y" → (X, wrote, Y)
    - "X is in Y" → (X, located_in, Y)
    - "X is the capital of Y" → (Y, capital_of, X)
    """
    
    # Relation patterns: (regex, subject_group, relation, object_group)
    PATTERNS = [
        # "Paris is the capital of France"
        (r"(\w+)\s+is\s+the\s+capital\s+of\s+(\w+)", 2, "capital_of", 1),
        
        # "The capital of France is Paris"
        (r"the\s+capital\s+of\s+(\w+)\s+is\s+(\w+)", 1, "capital_of", 2),
        
        # "X wrote Y" / "X authored Y"
        (r"(\w+)\s+(?:wrote|authored|created)\s+(.+?)(?:\.|$)", 1, "wrote", 2),
        
        # "Y was written by X"
        (r"(.+?)\s+was\s+(?:written|authored|created)\s+by\s+(\w+)", 2, "wrote", 1),
        
        # "X is located in Y" / "X is in Y"
        (r"(\w+)\s+is\s+(?:located\s+)?in\s+(\w+)", 1, "located_in", 2),
        
        # "X is a Y" / "X is an Y"
        (r"(\w+)\s+is\s+(?:a|an)\s+(\w+)", 1, "is_a", 2),
        
        # "X born in Y" (year or place)
        (r"(\w+)\s+(?:was\s+)?born\s+in\s+(\w+)", 1, "born_in", 2),
        
        # "X died in Y"
        (r"(\w+)\s+(?:died|passed away)\s+in\s+(\w+)", 1, "died_in", 2),
        
        # "X founded Y"
        (r"(\w+)\s+founded\s+(\w+)", 1, "founded", 2),
        
        # "X leads Y" / "X is the leader of Y"
        (r"(\w+)\s+(?:leads|is\s+the\s+leader\s+of)\s+(\w+)", 1, "leads", 2),
        
        # "X contains Y"
        (r"(\w+)\s+contains\s+(\w+)", 1, "contains", 2),
        
        # Generic "X verb Y" fallback
        (r"(\w+)\s+(\w+(?:s|ed|ing)?)\s+(\w+)", 1, None, 3),  # verb in group 2
    ]
    
    # Relation inverses
    INVERSES = {
        "capital_of": "has_capital",
        "located_in": "contains",
        "wrote": "written_by",
        "is_a": "has_instance",
        "born_in": "birthplace_of",
        "founded": "founded_by",
        "leads": "led_by",
        "contains": "located_in",
    }
    
    def __init__(self):
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), subj, rel, obj)
            for pattern, subj, rel, obj in self.PATTERNS
        ]
    
    def parse(self, text: str) -> List[Fact]:
        """Parse text into facts."""
        facts = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            fact = self._parse_sentence(sentence)
            if fact:
                facts.append(fact)
        
        return facts
    
    def _parse_sentence(self, sentence: str) -> Optional[Fact]:
        """Parse a single sentence into a fact."""
        for pattern, subj_group, relation, obj_group in self.compiled_patterns:
            match = pattern.search(sentence)
            if match:
                try:
                    subject = match.group(subj_group).strip().lower()
                    object_ = match.group(obj_group).strip().lower()
                    
                    # Handle generic verb pattern
                    if relation is None:
                        relation = match.group(2).strip().lower()
                    
                    # Clean up
                    subject = self._clean_entity(subject)
                    object_ = self._clean_entity(object_)
                    
                    if subject and object_ and subject != object_:
                        return Fact(subject, relation, object_, source=sentence)
                except:
                    continue
        
        return None
    
    def _clean_entity(self, entity: str) -> str:
        """Clean entity name."""
        # Remove articles
        entity = re.sub(r'^(the|a|an)\s+', '', entity)
        # Remove quotes
        entity = entity.strip('"\'')
        # Replace spaces with underscores
        entity = entity.replace(' ', '_')
        return entity


# =============================================================================
# MULTI-HOP REASONING
# =============================================================================

class ReasoningEngine:
    """
    Multi-hop reasoning over the geometric space.
    
    Supports:
    - Path queries: A --r1--> ? --r2--> B
    - Transitive closure: A --r*--> B
    - Inverse relations: A <--r-- B
    """
    
    def __init__(self, lcm: 'GeometricLCM'):
        self.lcm = lcm
    
    def multi_hop_query(self, start: str, relations: List[str], 
                        k: int = 5) -> List[Tuple[str, float, List[str]]]:
        """
        Execute multi-hop query: start --r1--> ? --r2--> ? ...
        
        Returns: [(final_entity, confidence, path), ...]
        """
        if not relations:
            return [(start, 1.0, [start])]
        
        # Start with initial entity
        current_results = [(start, 1.0, [start])]
        
        for relation in relations:
            next_results = []
            
            for entity, conf, path in current_results:
                # Query this hop
                hop_results = self.lcm.query(entity, relation, k=k)
                
                for next_entity, sim in hop_results:
                    new_conf = conf * sim
                    new_path = path + [f"--{relation}-->", next_entity]
                    next_results.append((next_entity, new_conf, new_path))
            
            # Keep top k
            next_results.sort(key=lambda x: -x[1])
            current_results = next_results[:k]
        
        return current_results
    
    def find_path(self, start: str, end: str, max_hops: int = 3,
                  k: int = 5) -> List[Tuple[List[str], float]]:
        """
        Find paths from start to end entity.
        
        Returns: [(path, confidence), ...]
        """
        if start not in self.lcm.entities or end not in self.lcm.entities:
            return []
        
        # BFS with beam search
        paths = [([start], 1.0)]
        found_paths = []
        
        for hop in range(max_hops):
            next_paths = []
            
            for path, conf in paths:
                current = path[-1]
                
                # Try each relation
                for rel_name, relation in self.lcm.relations.items():
                    results = self.lcm.query(current, rel_name, k=k)
                    
                    for next_entity, sim in results:
                        if next_entity in path:  # Avoid cycles
                            continue
                        
                        new_path = path + [rel_name, next_entity]
                        new_conf = conf * sim
                        
                        if next_entity == end:
                            found_paths.append((new_path, new_conf))
                        else:
                            next_paths.append((new_path, new_conf))
            
            # Keep top k paths
            next_paths.sort(key=lambda x: -x[1])
            paths = next_paths[:k * 2]
        
        found_paths.sort(key=lambda x: -x[1])
        return found_paths[:k]
    
    def transitive_query(self, start: str, relation: str, 
                         max_hops: int = 5, k: int = 10) -> List[Tuple[str, float, int]]:
        """
        Transitive closure: find all entities reachable via relation.
        
        Returns: [(entity, confidence, hops), ...]
        """
        visited = {start: (1.0, 0)}
        frontier = [(start, 1.0, 0)]
        
        while frontier:
            current, conf, hops = frontier.pop(0)
            
            if hops >= max_hops:
                continue
            
            results = self.lcm.query(current, relation, k=k)
            
            for next_entity, sim in results:
                new_conf = conf * sim
                new_hops = hops + 1
                
                if next_entity not in visited or visited[next_entity][0] < new_conf:
                    visited[next_entity] = (new_conf, new_hops)
                    frontier.append((next_entity, new_conf, new_hops))
        
        # Remove start
        del visited[start]
        
        # Sort by confidence
        results = [(e, c, h) for e, (c, h) in visited.items()]
        results.sort(key=lambda x: -x[1])
        return results[:k]


# =============================================================================
# GEOMETRIC LCM (FULL VERSION)
# =============================================================================

class GeometricLCM:
    """
    Full Geometric Language Model.
    
    Features:
    - Natural language input
    - Dynamic learning
    - Multi-hop reasoning
    - Persistence
    """
    
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.facts: List[Fact] = []
        self.type_hints: Dict[str, Set[str]] = defaultdict(set)
        
        # Components
        self.parser = NLParser()
        self.reasoner = ReasoningEngine(self)
    
    # =========================================================================
    # ENTITY MANAGEMENT
    # =========================================================================
    
    def get_entity(self, name: str, create: bool = True) -> Optional[Entity]:
        """Get or create entity."""
        name = name.lower().strip()
        
        if name in self.entities:
            return self.entities[name]
        
        # Check aliases
        for entity in self.entities.values():
            if name in entity.aliases:
                return entity
        
        if not create:
            return None
        
        # Create new
        seed = hash(name) % (2**32)
        rng = np.random.default_rng(seed)
        pos = rng.standard_normal(self.dim)
        pos = pos / np.linalg.norm(pos)
        
        self.entities[name] = Entity(name=name, position=pos)
        return self.entities[name]
    
    def add_alias(self, entity_name: str, alias: str):
        """Add alias for entity."""
        entity = self.get_entity(entity_name)
        if entity:
            entity.aliases.add(alias.lower())
    
    def set_type(self, entity_name: str, entity_type: str):
        """Set entity type."""
        entity = self.get_entity(entity_name)
        if entity:
            entity.entity_type = entity_type
            self.type_hints[entity_type].add(entity_name)
    
    # =========================================================================
    # RELATION MANAGEMENT
    # =========================================================================
    
    def get_relation(self, name: str) -> Relation:
        """Get or create relation."""
        name = name.lower().strip()
        
        if name not in self.relations:
            seed = hash(f"__REL__{name}") % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim)
            vec = vec / np.linalg.norm(vec)
            
            # Check for inverse
            inverse = NLParser.INVERSES.get(name)
            
            self.relations[name] = Relation(
                name=name,
                vector=vec,
                inverse_name=inverse
            )
        
        return self.relations[name]
    
    # =========================================================================
    # FACT MANAGEMENT
    # =========================================================================
    
    def add_fact(self, subject: str, relation: str, object_: str,
                 subject_type: str = None, object_type: str = None,
                 source: str = ""):
        """Add a fact."""
        self.get_entity(subject)
        self.get_entity(object_)
        self.get_relation(relation)
        
        self.facts.append(Fact(subject, relation, object_, source=source))
        
        if subject_type:
            self.set_type(subject, subject_type)
        if object_type:
            self.set_type(object_, object_type)
    
    def ingest_text(self, text: str) -> List[Fact]:
        """Parse text and add facts."""
        facts = self.parser.parse(text)
        
        for fact in facts:
            self.add_fact(fact.subject, fact.relation, fact.object, 
                         source=fact.source)
        
        return facts
    
    # =========================================================================
    # LEARNING
    # =========================================================================
    
    def learn(self, n_iterations: int = 100, target_consistency: float = 0.95,
              verbose: bool = False) -> float:
        """Learn entity positions and relation vectors."""
        if not self.facts:
            return 1.0
        
        # Group facts by relation
        facts_by_relation = defaultdict(list)
        for fact in self.facts:
            facts_by_relation[fact.relation].append((fact.subject, fact.object))
        
        for iteration in range(n_iterations):
            lr = 0.3 * (1.0 - iteration / n_iterations)
            
            # Update relation vectors
            for rel_name, pairs in facts_by_relation.items():
                self._update_relation(rel_name, pairs)
            
            # Update entity positions
            for fact in self.facts:
                self._update_positions(fact.subject, fact.relation, fact.object, lr)
            
            # Check consistency
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
    
    def _update_relation(self, rel_name: str, pairs: List[Tuple[str, str]]):
        """Update relation vector from observed pairs."""
        offsets = []
        for subj, obj in pairs:
            if subj in self.entities and obj in self.entities:
                offset = self.entities[obj].position - self.entities[subj].position
                norm = np.linalg.norm(offset)
                if norm > 1e-10:
                    offsets.append(offset / norm)
        
        if offsets:
            avg = np.mean(offsets, axis=0)
            avg = avg / np.linalg.norm(avg)
            
            rel = self.relations[rel_name]
            rel.vector = 0.7 * avg + 0.3 * rel.vector
            rel.vector = rel.vector / np.linalg.norm(rel.vector)
            rel.instance_count = len(pairs)
    
    def _update_positions(self, subj: str, rel: str, obj: str, lr: float):
        """Update entity positions to align with relation."""
        subj_entity = self.entities[subj]
        obj_entity = self.entities[obj]
        rel_vec = self.relations[rel].vector
        
        # Object should be at subject + relation
        target_obj = subj_entity.position + rel_vec
        target_obj = target_obj / np.linalg.norm(target_obj)
        obj_entity.position = (1 - lr) * obj_entity.position + lr * target_obj
        obj_entity.position = obj_entity.position / np.linalg.norm(obj_entity.position)
        
        # Subject should be at object - relation
        target_subj = obj_entity.position - rel_vec
        target_subj = target_subj / np.linalg.norm(target_subj)
        subj_entity.position = (1 - lr*0.5) * subj_entity.position + lr*0.5 * target_subj
        subj_entity.position = subj_entity.position / np.linalg.norm(subj_entity.position)
    
    def _compute_consistency(self, rel_name: str) -> float:
        """Compute relation consistency."""
        pairs = [(f.subject, f.object) for f in self.facts if f.relation == rel_name]
        
        if len(pairs) < 2:
            return 1.0
        
        offsets = []
        for subj, obj in pairs:
            if subj in self.entities and obj in self.entities:
                offset = self.entities[obj].position - self.entities[subj].position
                norm = np.linalg.norm(offset)
                if norm > 1e-10:
                    offsets.append(offset / norm)
        
        if len(offsets) < 2:
            return 1.0
        
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
        subject = subject.lower()
        relation = relation.lower()
        
        if subject not in self.entities or relation not in self.relations:
            return []
        
        target = self.entities[subject].position + self.relations[relation].vector
        target = target / np.linalg.norm(target)
        
        results = []
        for name, entity in self.entities.items():
            if name == subject:
                continue
            sim = np.dot(target, entity.position)
            results.append((name, float(sim)))
        
        return sorted(results, key=lambda x: -x[1])[:k]
    
    def inverse_query(self, object_: str, relation: str, k: int = 5) -> List[Tuple[str, float]]:
        """Query: ? --relation--> object"""
        object_ = object_.lower()
        relation = relation.lower()
        
        if object_ not in self.entities or relation not in self.relations:
            return []
        
        target = self.entities[object_].position - self.relations[relation].vector
        target = target / np.linalg.norm(target)
        
        results = []
        for name, entity in self.entities.items():
            if name == object_:
                continue
            sim = np.dot(target, entity.position)
            results.append((name, float(sim)))
        
        return sorted(results, key=lambda x: -x[1])[:k]
    
    def analogy(self, a: str, b: str, c: str, k: int = 5) -> List[Tuple[str, float]]:
        """Solve: a:b :: c:?"""
        a, b, c = a.lower(), b.lower(), c.lower()
        
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
            results.append((name, float(sim)))
        
        return sorted(results, key=lambda x: -x[1])[:k]
    
    def similar(self, entity: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find similar entities."""
        entity = entity.lower()
        
        if entity not in self.entities:
            return []
        
        pos = self.entities[entity].position
        
        results = []
        for name, e in self.entities.items():
            if name == entity:
                continue
            sim = np.dot(pos, e.position)
            results.append((name, float(sim)))
        
        return sorted(results, key=lambda x: -x[1])[:k]
    
    # =========================================================================
    # NATURAL LANGUAGE INTERFACE
    # =========================================================================
    
    def ask(self, question: str) -> str:
        """Answer a natural language question."""
        question = question.lower().strip()
        
        # Pattern: "What is the X of Y?"
        match = re.search(r"what\s+is\s+the\s+(\w+)\s+of\s+(\w+)", question)
        if match:
            relation = match.group(1)
            subject = match.group(2)
            results = self.query(subject, relation, k=1)
            if results:
                return f"The {relation} of {subject} is {results[0][0]}."
            return f"I don't know the {relation} of {subject}."
        
        # Pattern: "Who wrote X?"
        match = re.search(r"who\s+wrote\s+(.+?)[\?]?$", question)
        if match:
            book = match.group(1).strip()
            results = self.inverse_query(book, "wrote", k=1)
            if results:
                return f"{results[0][0].title()} wrote {book}."
            return f"I don't know who wrote {book}."
        
        # Pattern: "Where is X?"
        match = re.search(r"where\s+is\s+(\w+)", question)
        if match:
            entity = match.group(1)
            results = self.query(entity, "located_in", k=1)
            if results:
                return f"{entity.title()} is in {results[0][0]}."
            return f"I don't know where {entity} is."
        
        # Pattern: "X is to Y as Z is to what?"
        match = re.search(r"(\w+)\s+is\s+to\s+(\w+)\s+as\s+(\w+)\s+is\s+to\s+(?:what|\?)", question)
        if match:
            a, b, c = match.groups()
            results = self.analogy(a, b, c, k=1)
            if results:
                return f"{a.title()} is to {b} as {c} is to {results[0][0]}."
            return f"I can't solve that analogy."
        
        return "I don't understand that question."
    
    def tell(self, statement: str) -> str:
        """Learn from a natural language statement."""
        facts = self.ingest_text(statement)
        
        if facts:
            self.learn(n_iterations=20, verbose=False)
            fact_strs = [f"{f.subject} --{f.relation}--> {f.object}" for f in facts]
            return f"Learned: {', '.join(fact_strs)}"
        
        return "I couldn't extract any facts from that statement."
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save(self, filepath: str):
        """Save to file."""
        data = {
            'dim': self.dim,
            'entities': {
                name: {
                    'position': e.position.tolist(),
                    'entity_type': e.entity_type,
                    'aliases': list(e.aliases),
                    'metadata': e.metadata
                }
                for name, e in self.entities.items()
            },
            'relations': {
                name: {
                    'vector': r.vector.tolist(),
                    'inverse_name': r.inverse_name,
                    'consistency': r.consistency,
                    'instance_count': r.instance_count
                }
                for name, r in self.relations.items()
            },
            'facts': [
                {'subject': f.subject, 'relation': f.relation, 
                 'object': f.object, 'confidence': f.confidence, 'source': f.source}
                for f in self.facts
            ],
            'type_hints': {k: list(v) for k, v in self.type_hints.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.dim = data['dim']
        
        self.entities = {
            name: Entity(
                name=name,
                position=np.array(e['position']),
                entity_type=e['entity_type'],
                aliases=set(e.get('aliases', [])),
                metadata=e.get('metadata', {})
            )
            for name, e in data['entities'].items()
        }
        
        self.relations = {
            name: Relation(
                name=name,
                vector=np.array(r['vector']),
                inverse_name=r.get('inverse_name'),
                consistency=r.get('consistency', 0),
                instance_count=r.get('instance_count', 0)
            )
            for name, r in data['relations'].items()
        }
        
        self.facts = [
            Fact(f['subject'], f['relation'], f['object'], 
                 f.get('confidence', 1.0), f.get('source', ''))
            for f in data['facts']
        ]
        
        self.type_hints = {k: set(v) for k, v in data.get('type_hints', {}).items()}
    
    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================
    
    def status(self) -> Dict:
        """Get system status."""
        return {
            'entities': len(self.entities),
            'relations': len(self.relations),
            'facts': len(self.facts),
            'types': {t: len(e) for t, e in self.type_hints.items()},
            'relation_consistency': {
                r.name: r.consistency for r in self.relations.values()
            }
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo_full_system():
    """Demonstrate the full Geometric LCM."""
    
    print("=" * 70)
    print("GEOMETRIC LCM - FULL DEMONSTRATION")
    print("=" * 70)
    print()
    
    lcm = GeometricLCM(dim=256)
    
    # =========================================================================
    # PART 1: Natural Language Ingestion
    # =========================================================================
    
    print("PART 1: Natural Language Ingestion")
    print("-" * 50)
    
    texts = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Tokyo is the capital of Japan.",
        "Rome is the capital of Italy.",
        "Madrid is the capital of Spain.",
        "London is the capital of UK.",
        "Beijing is the capital of China.",
        "Moscow is the capital of Russia.",
        "Melville wrote Moby Dick.",
        "Shakespeare wrote Hamlet.",
        "Orwell wrote 1984.",
        "Tolkien wrote Lord of the Rings.",
        "The Eiffel Tower is in Paris.",
        "The Colosseum is in Rome.",
        "Big Ben is in London.",
    ]
    
    for text in texts:
        result = lcm.tell(text)
        print(f"  {result}")
    
    print()
    print(f"Status: {lcm.status()}")
    print()
    
    # =========================================================================
    # PART 2: Learning
    # =========================================================================
    
    print("PART 2: Learning")
    print("-" * 50)
    
    lcm.learn(n_iterations=100, target_consistency=0.95, verbose=True)
    
    print()
    print("Relation consistencies:")
    for name, rel in lcm.relations.items():
        print(f"  {name}: {rel.consistency:.3f} ({rel.instance_count} instances)")
    print()
    
    # =========================================================================
    # PART 3: Queries
    # =========================================================================
    
    print("PART 3: Queries")
    print("-" * 50)
    
    queries = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "Who wrote Hamlet?",
        "Where is the Eiffel Tower?",
    ]
    
    for q in queries:
        answer = lcm.ask(q)
        print(f"  Q: {q}")
        print(f"  A: {answer}")
        print()
    
    # =========================================================================
    # PART 4: Analogies
    # =========================================================================
    
    print("PART 4: Analogies")
    print("-" * 50)
    
    analogies = [
        ("france", "paris", "germany", "berlin"),
        ("france", "paris", "japan", "tokyo"),
        ("france", "paris", "china", "beijing"),
        ("melville", "moby_dick", "shakespeare", "hamlet"),
    ]
    
    correct = 0
    for a, b, c, expected in analogies:
        results = lcm.analogy(a, b, c, k=1)
        answer = results[0][0] if results else "?"
        match = "✓" if answer == expected else "✗"
        print(f"  {match} {a}:{b} :: {c}:? → {answer} (expected: {expected})")
        if answer == expected:
            correct += 1
    
    print(f"\nAnalogy accuracy: {correct}/{len(analogies)} = {correct/len(analogies):.1%}")
    print()
    
    # =========================================================================
    # PART 5: Multi-Hop Reasoning
    # =========================================================================
    
    print("PART 5: Multi-Hop Reasoning")
    print("-" * 50)
    
    # Add some more facts for multi-hop
    lcm.tell("France is in Europe.")
    lcm.tell("Germany is in Europe.")
    lcm.tell("Japan is in Asia.")
    lcm.tell("China is in Asia.")
    lcm.learn(n_iterations=30, verbose=False)
    
    print("Multi-hop: france --capital_of--> ? --located_in--> ?")
    results = lcm.reasoner.multi_hop_query("france", ["capital_of", "located_in"], k=3)
    for entity, conf, path in results[:3]:
        path_str = " ".join(path)
        print(f"  {path_str} (conf: {conf:.3f})")
    
    print()
    
    print("Path finding: france → europe")
    paths = lcm.reasoner.find_path("france", "europe", max_hops=2, k=3)
    for path, conf in paths[:3]:
        path_str = " → ".join(path)
        print(f"  {path_str} (conf: {conf:.3f})")
    
    print()
    
    # =========================================================================
    # PART 6: Incremental Learning
    # =========================================================================
    
    print("PART 6: Incremental Learning")
    print("-" * 50)
    
    new_facts = [
        "Brasilia is the capital of Brazil.",
        "Delhi is the capital of India.",
        "Hemingway wrote The Old Man and the Sea.",
    ]
    
    for text in new_facts:
        result = lcm.tell(text)
        print(f"  {result}")
    
    print()
    
    # Test new analogies
    print("Testing new analogies:")
    new_analogies = [
        ("france", "paris", "brazil", "brasilia"),
        ("france", "paris", "india", "delhi"),
    ]
    
    correct = 0
    for a, b, c, expected in new_analogies:
        results = lcm.analogy(a, b, c, k=1)
        answer = results[0][0] if results else "?"
        match = "✓" if answer == expected else "✗"
        print(f"  {match} {a}:{b} :: {c}:? → {answer} (expected: {expected})")
        if answer == expected:
            correct += 1
    
    print(f"\nNew analogy accuracy: {correct}/{len(new_analogies)} = {correct/len(new_analogies):.1%}")
    print()
    
    # =========================================================================
    # PART 7: Persistence
    # =========================================================================
    
    print("PART 7: Persistence")
    print("-" * 50)
    
    save_path = "/tmp/geometric_lcm_demo.json"
    lcm.save(save_path)
    print(f"  Saved to {save_path}")
    
    # Load into new instance
    lcm2 = GeometricLCM()
    lcm2.load(save_path)
    print(f"  Loaded: {lcm2.status()}")
    
    # Verify it works
    answer = lcm2.ask("What is the capital of France?")
    print(f"  Verification: {answer}")
    
    print()
    
    return lcm


def demo_scaling():
    """Test scaling with more data."""
    
    print()
    print("=" * 70)
    print("SCALING TEST")
    print("=" * 70)
    print()
    
    lcm = GeometricLCM(dim=256)
    
    # Generate synthetic data
    import time
    
    n_countries = 50
    n_authors = 30
    
    print(f"Generating {n_countries} countries and {n_authors} authors...")
    
    # Countries and capitals
    for i in range(n_countries):
        country = f"country_{i}"
        capital = f"capital_{i}"
        lcm.add_fact(country, "capital_of", capital)
    
    # Authors and books
    for i in range(n_authors):
        author = f"author_{i}"
        book = f"book_{i}"
        lcm.add_fact(author, "wrote", book)
    
    print(f"Total facts: {len(lcm.facts)}")
    print()
    
    # Time learning
    print("Learning...")
    start = time.time()
    lcm.learn(n_iterations=100, target_consistency=0.95, verbose=True)
    elapsed = time.time() - start
    print(f"Learning time: {elapsed:.2f}s")
    print()
    
    # Test accuracy
    print("Testing accuracy...")
    
    correct = 0
    total = min(20, n_countries)
    
    for i in range(total):
        results = lcm.query(f"country_{i}", "capital_of", k=1)
        if results and results[0][0] == f"capital_{i}":
            correct += 1
    
    print(f"Query accuracy: {correct}/{total} = {correct/total:.1%}")
    
    # Test analogies
    correct = 0
    total = min(10, n_countries - 1)
    
    for i in range(total):
        results = lcm.analogy(f"country_0", f"capital_0", f"country_{i+1}", k=1)
        if results and results[0][0] == f"capital_{i+1}":
            correct += 1
    
    print(f"Analogy accuracy: {correct}/{total} = {correct/total:.1%}")
    print()
    
    return lcm


if __name__ == "__main__":
    lcm = demo_full_system()
    demo_scaling()
