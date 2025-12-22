"""
Geometric LCM - Core Module

A dynamic geometric language model that integrates with TruthSpace.

Features:
- Natural language parsing to facts
- Dynamic learning (structure IS data)
- Multi-hop reasoning
- Integration with TruthSpace Vocabulary and KnowledgeBase

This is the production version of the experimental geometric_lcm_full.py
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json

from .vocabulary import Vocabulary, cosine_similarity, tokenize


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GeoEntity:
    """An entity in the geometric space."""
    name: str
    position: np.ndarray
    entity_type: str = "unknown"
    aliases: Set[str] = field(default_factory=set)
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class GeoRelation:
    """A relation between entities - stored as a learned vector offset."""
    name: str
    vector: np.ndarray
    inverse_name: Optional[str] = None
    consistency: float = 0.0
    instance_count: int = 0


@dataclass
class GeoFact:
    """A fact: subject --relation--> object"""
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    source: str = ""


# =============================================================================
# NATURAL LANGUAGE PARSER
# =============================================================================

class FactParser:
    """Parse natural language into facts."""
    
    PATTERNS = [
        # "X is the capital of Y"
        (r"(\w+)\s+is\s+the\s+capital\s+of\s+(\w+)", 2, "capital_of", 1),
        (r"the\s+capital\s+of\s+(\w+)\s+is\s+(\w+)", 1, "capital_of", 2),
        
        # "X wrote/authored Y"
        (r"(\w+)\s+(?:wrote|authored|created)\s+(.+?)(?:\.|$)", 1, "wrote", 2),
        (r"(.+?)\s+was\s+(?:written|authored)\s+by\s+(\w+)", 2, "wrote", 1),
        
        # "X is in Y" / "X is located in Y"
        (r"(\w+)\s+is\s+(?:located\s+)?in\s+(\w+)", 1, "located_in", 2),
        
        # "X is a/an Y"
        (r"(\w+)\s+is\s+(?:a|an)\s+(\w+)", 1, "is_a", 2),
        
        # "X founded Y"
        (r"(\w+)\s+founded\s+(\w+)", 1, "founded", 2),
        
        # "X leads Y"
        (r"(\w+)\s+(?:leads|is\s+the\s+leader\s+of)\s+(\w+)", 1, "leads", 2),
        
        # "X contains Y"
        (r"(\w+)\s+contains\s+(\w+)", 1, "contains", 2),
        
        # "X born in Y"
        (r"(\w+)\s+(?:was\s+)?born\s+in\s+(\w+)", 1, "born_in", 2),
    ]
    
    INVERSES = {
        "capital_of": "has_capital",
        "located_in": "contains",
        "wrote": "written_by",
        "is_a": "has_instance",
        "founded": "founded_by",
        "leads": "led_by",
        "contains": "located_in",
        "born_in": "birthplace_of",
    }
    
    def __init__(self):
        self.compiled = [
            (re.compile(p, re.IGNORECASE), s, r, o)
            for p, s, r, o in self.PATTERNS
        ]
    
    def parse(self, text: str) -> List[GeoFact]:
        """Parse text into facts."""
        facts = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            for pattern, subj_g, rel, obj_g in self.compiled:
                match = pattern.search(sentence)
                if match:
                    try:
                        subj = self._clean(match.group(subj_g))
                        obj = self._clean(match.group(obj_g))
                        
                        if subj and obj and subj != obj:
                            facts.append(GeoFact(subj, rel, obj, source=sentence))
                            break
                    except:
                        continue
        
        return facts
    
    def _clean(self, text: str) -> str:
        """Clean entity name."""
        text = re.sub(r'^(the|a|an)\s+', '', text.lower())
        text = text.strip('"\'')
        text = re.sub(r'\s+', '_', text)
        return text


# =============================================================================
# GEOMETRIC LCM
# =============================================================================

class GeometricLCM:
    """
    Dynamic Geometric Language Model.
    
    Integrates with TruthSpace Vocabulary for text encoding while maintaining
    its own geometric space for relational reasoning.
    
    Key principles:
    1. Structure IS the data - positions and relations encode knowledge
    2. Learning IS structure update - new facts modify geometry
    3. Inference IS geometric - queries are projections and similarities
    """
    
    def __init__(self, dim: int = 256, vocab: Vocabulary = None):
        self.dim = dim
        self.vocab = vocab or Vocabulary(dim=dim)
        
        self.entities: Dict[str, GeoEntity] = {}
        self.relations: Dict[str, GeoRelation] = {}
        self.facts: List[GeoFact] = []
        self.type_hints: Dict[str, Set[str]] = defaultdict(set)
        
        self.parser = FactParser()
    
    # =========================================================================
    # ENTITY MANAGEMENT
    # =========================================================================
    
    def get_entity(self, name: str, create: bool = True) -> Optional[GeoEntity]:
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
        
        # Use vocabulary's deterministic positioning
        position = self.vocab.get_position(name)
        
        self.entities[name] = GeoEntity(name=name, position=position.copy())
        return self.entities[name]
    
    def set_type(self, name: str, entity_type: str):
        """Set entity type."""
        entity = self.get_entity(name)
        if entity:
            entity.entity_type = entity_type
            self.type_hints[entity_type].add(name)
    
    def add_alias(self, name: str, alias: str):
        """Add alias for entity."""
        entity = self.get_entity(name)
        if entity:
            entity.aliases.add(alias.lower())
    
    # =========================================================================
    # RELATION MANAGEMENT
    # =========================================================================
    
    def get_relation(self, name: str) -> GeoRelation:
        """Get or create relation."""
        name = name.lower()
        
        if name not in self.relations:
            # Initialize with deterministic vector
            vec = self.vocab.get_position(f"__relation__{name}")
            inverse = FactParser.INVERSES.get(name)
            
            self.relations[name] = GeoRelation(
                name=name,
                vector=vec.copy(),
                inverse_name=inverse
            )
        
        return self.relations[name]
    
    # =========================================================================
    # FACT MANAGEMENT
    # =========================================================================
    
    def add_fact(self, subject: str, relation: str, object_: str,
                 subject_type: str = None, object_type: str = None) -> GeoFact:
        """Add a fact."""
        self.get_entity(subject)
        self.get_entity(object_)
        self.get_relation(relation)
        
        fact = GeoFact(subject.lower(), relation.lower(), object_.lower())
        self.facts.append(fact)
        
        if subject_type:
            self.set_type(subject, subject_type)
        if object_type:
            self.set_type(object_, object_type)
        
        return fact
    
    def ingest(self, text: str) -> List[GeoFact]:
        """Parse text and add facts."""
        facts = self.parser.parse(text)
        
        for fact in facts:
            self.add_fact(fact.subject, fact.relation, fact.object)
        
        return facts
    
    # =========================================================================
    # LEARNING
    # =========================================================================
    
    def learn(self, n_iterations: int = 100, target_consistency: float = 0.95,
              verbose: bool = False) -> float:
        """
        Learn entity positions and relation vectors from facts.
        
        This is the core "structure update" operation.
        """
        if not self.facts:
            return 1.0
        
        # Group facts by relation
        by_relation = defaultdict(list)
        for fact in self.facts:
            by_relation[fact.relation].append((fact.subject, fact.object))
        
        for iteration in range(n_iterations):
            lr = 0.3 * (1.0 - iteration / n_iterations)
            
            # Update relation vectors from current positions
            for rel_name, pairs in by_relation.items():
                self._update_relation(rel_name, pairs)
            
            # Update entity positions to align with relations
            for fact in self.facts:
                self._update_positions(fact.subject, fact.relation, fact.object, lr)
            
            # Check consistency
            min_consistency = 1.0
            for rel_name in by_relation:
                consistency = self._compute_consistency(rel_name)
                self.relations[rel_name].consistency = consistency
                min_consistency = min(min_consistency, consistency)
            
            if verbose and iteration % 10 == 0:
                print(f"  Iteration {iteration}: consistency = {min_consistency:.3f}")
            
            if min_consistency >= target_consistency:
                if verbose:
                    print(f"  Converged at iteration {iteration}")
                break
        
        return min_consistency
    
    def _update_relation(self, rel_name: str, pairs: List[Tuple[str, str]]):
        """Update relation vector as average of observed offsets."""
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
        subj_e = self.entities[subj]
        obj_e = self.entities[obj]
        rel_v = self.relations[rel].vector
        
        # Object should be at subject + relation
        target_obj = subj_e.position + rel_v
        target_obj = target_obj / np.linalg.norm(target_obj)
        obj_e.position = (1 - lr) * obj_e.position + lr * target_obj
        obj_e.position = obj_e.position / np.linalg.norm(obj_e.position)
        
        # Subject should be at object - relation
        target_subj = obj_e.position - rel_v
        target_subj = target_subj / np.linalg.norm(target_subj)
        subj_e.position = (1 - lr*0.5) * subj_e.position + lr*0.5 * target_subj
        subj_e.position = subj_e.position / np.linalg.norm(subj_e.position)
    
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
        
        return float(np.mean(sims))
    
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
            sim = float(np.dot(target, entity.position))
            results.append((name, sim))
        
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
            sim = float(np.dot(target, entity.position))
            results.append((name, sim))
        
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
            sim = float(np.dot(target, entity.position))
            results.append((name, sim))
        
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
            sim = float(np.dot(pos, e.position))
            results.append((name, sim))
        
        return sorted(results, key=lambda x: -x[1])[:k]
    
    # =========================================================================
    # MULTI-HOP REASONING
    # =========================================================================
    
    def multi_hop(self, start: str, relations: List[str], 
                  k: int = 5) -> List[Tuple[str, float, List[str]]]:
        """
        Multi-hop query: start --r1--> ? --r2--> ? ...
        
        Returns: [(final_entity, confidence, path), ...]
        """
        current = [(start.lower(), 1.0, [start.lower()])]
        
        for rel in relations:
            next_results = []
            
            for entity, conf, path in current:
                hop_results = self.query(entity, rel, k=k)
                
                for next_e, sim in hop_results:
                    new_conf = conf * sim
                    new_path = path + [f"--{rel}-->", next_e]
                    next_results.append((next_e, new_conf, new_path))
            
            next_results.sort(key=lambda x: -x[1])
            current = next_results[:k]
        
        return current
    
    def find_path(self, start: str, end: str, max_hops: int = 3,
                  k: int = 5) -> List[Tuple[List[str], float]]:
        """Find paths from start to end."""
        start, end = start.lower(), end.lower()
        
        if start not in self.entities or end not in self.entities:
            return []
        
        paths = [([start], 1.0)]
        found = []
        
        for _ in range(max_hops):
            next_paths = []
            
            for path, conf in paths:
                current = path[-1]
                
                for rel_name in self.relations:
                    results = self.query(current, rel_name, k=k)
                    
                    for next_e, sim in results:
                        if next_e in path:
                            continue
                        
                        new_path = path + [rel_name, next_e]
                        new_conf = conf * sim
                        
                        if next_e == end:
                            found.append((new_path, new_conf))
                        else:
                            next_paths.append((new_path, new_conf))
            
            next_paths.sort(key=lambda x: -x[1])
            paths = next_paths[:k * 2]
        
        found.sort(key=lambda x: -x[1])
        return found[:k]
    
    # =========================================================================
    # NATURAL LANGUAGE INTERFACE
    # =========================================================================
    
    def ask(self, question: str) -> str:
        """Answer a natural language question."""
        q = question.lower().strip()
        
        # "What is the X of Y?"
        match = re.search(r"what\s+(?:is\s+)?the\s+(\w+)\s+of\s+(\w+)", q)
        if match:
            rel, subj = match.groups()
            # Map common terms to relation names
            rel_map = {"capital": "capital_of", "author": "wrote"}
            rel = rel_map.get(rel, rel)
            
            results = self.query(subj, rel, k=1)
            if results and results[0][1] > 0.5:
                return f"The {match.group(1)} of {subj} is {results[0][0]}."
            return f"I don't know the {match.group(1)} of {subj}."
        
        # "Who wrote X?"
        match = re.search(r"who\s+wrote\s+(.+?)[\?]?$", q)
        if match:
            book_raw = match.group(1).strip()
            book = book_raw.replace(' ', '_')
            
            # Try exact match first
            results = self.inverse_query(book, "wrote", k=1)
            if results and results[0][1] > 0.5:
                return f"{results[0][0].replace('_', ' ').title()} wrote {book_raw}."
            
            # Try fuzzy match - find best matching book entity
            best_match = self._find_best_entity_match(book, entity_type="book")
            if best_match:
                results = self.inverse_query(best_match, "wrote", k=1)
                if results and results[0][1] > 0.5:
                    return f"{results[0][0].replace('_', ' ').title()} wrote {best_match.replace('_', ' ')}."
            
            return f"I don't know who wrote {book_raw}."
        
        # "Where is X?"
        match = re.search(r"where\s+is\s+(.+?)[\?]?$", q)
        if match:
            entity = match.group(1).strip().replace(' ', '_')
            results = self.query(entity, "located_in", k=1)
            if results and results[0][1] > 0.5:
                return f"{match.group(1).title()} is in {results[0][0]}."
            return f"I don't know where {match.group(1)} is."
        
        # "X is to Y as Z is to what?"
        match = re.search(r"(\w+)\s+is\s+to\s+(\w+)\s+as\s+(\w+)\s+is\s+to", q)
        if match:
            a, b, c = match.groups()
            results = self.analogy(a, b, c, k=1)
            if results:
                return f"{a.title()} is to {b} as {c} is to {results[0][0]}."
            return "I can't solve that analogy."
        
        return "I don't understand that question."
    
    def _find_best_entity_match(self, query: str, entity_type: str = None) -> Optional[str]:
        """Find the best matching entity name using substring matching."""
        query = query.lower().replace(' ', '_')
        query_words = set(query.split('_'))
        
        best_match = None
        best_score = 0
        
        for name in self.entities:
            # Check if query is substring of entity or vice versa
            if query in name or name in query:
                score = len(query) / max(len(name), len(query))
                if score > best_score:
                    best_score = score
                    best_match = name
                continue
            
            # Check word overlap
            name_words = set(name.split('_'))
            overlap = len(query_words & name_words)
            if overlap > 0:
                score = overlap / max(len(query_words), len(name_words))
                if score > best_score:
                    best_score = score
                    best_match = name
        
        return best_match if best_score > 0.3 else None
    
    def tell(self, statement: str) -> str:
        """Learn from a statement."""
        facts = self.ingest(statement)
        
        if facts:
            self.learn(n_iterations=30, verbose=False)
            fact_strs = [f"{f.subject} --{f.relation}--> {f.object}" for f in facts]
            return f"Learned: {', '.join(fact_strs)}"
        
        return "I couldn't extract any facts from that."
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save(self, filepath: str):
        """Save to JSON file."""
        data = {
            'dim': self.dim,
            'entities': {
                name: {
                    'position': e.position.tolist(),
                    'entity_type': e.entity_type,
                    'aliases': list(e.aliases)
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
                 'object': f.object, 'confidence': f.confidence}
                for f in self.facts
            ],
            'type_hints': {k: list(v) for k, v in self.type_hints.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.dim = data['dim']
        
        self.entities = {
            name: GeoEntity(
                name=name,
                position=np.array(e['position']),
                entity_type=e.get('entity_type', 'unknown'),
                aliases=set(e.get('aliases', []))
            )
            for name, e in data['entities'].items()
        }
        
        self.relations = {
            name: GeoRelation(
                name=name,
                vector=np.array(r['vector']),
                inverse_name=r.get('inverse_name'),
                consistency=r.get('consistency', 0),
                instance_count=r.get('instance_count', 0)
            )
            for name, r in data['relations'].items()
        }
        
        self.facts = [
            GeoFact(f['subject'], f['relation'], f['object'], f.get('confidence', 1.0))
            for f in data['facts']
        ]
        
        self.type_hints = defaultdict(set, {
            k: set(v) for k, v in data.get('type_hints', {}).items()
        })
    
    # =========================================================================
    # STATUS
    # =========================================================================
    
    def status(self) -> Dict:
        """Get system status."""
        return {
            'entities': len(self.entities),
            'relations': len(self.relations),
            'facts': len(self.facts),
            'types': dict(self.type_hints),
            'consistencies': {
                r.name: r.consistency for r in self.relations.values()
            }
        }
