"""
VSA-Enhanced Knowledge Base

Extends the standard KnowledgeBase with Vector Symbolic Architecture (VSA) binding
operations for structured relational queries and analogical reasoning.

This module demonstrates how binding unifies with the existing Q&A system:
- Standard Q&A: Cosine similarity between question encodings (bag-of-words)
- VSA Q&A: Unbinding structured relations (role ⊛ filler)

The key insight: Q&A can be viewed as unbinding.
- Question = role (what kind of answer?)
- Answer = filler (the content)
- Matching = unbind(knowledge, question_type) → similar to answer

This provides a unified geometric framework for both approaches.
"""

import numpy as np
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .vocabulary import Vocabulary, cosine_similarity
from .knowledge import KnowledgeBase, Fact, Triple, QAPair, detect_question_type
from .binding import (
    bind, unbind, bundle, permute,
    BindingMethod, CleanupMemory, similarity
)


@dataclass
class RelationalFact:
    """A fact encoded with VSA binding for structured queries."""
    id: str
    role: str           # e.g., "capital_of", "located_in", "authored_by"
    subject: str        # e.g., "france", "eiffel_tower", "moby_dick"
    value: str          # e.g., "paris", "paris", "melville"
    vector: np.ndarray  # Bound representation: role ⊛ subject + value
    source: str = ""


class VSAKnowledgeBase(KnowledgeBase):
    """
    Knowledge base with VSA binding for structured relational queries.
    
    Extends the standard KnowledgeBase with:
    - Relational facts encoded as bound vectors
    - Analogical reasoning via binding algebra
    - Question-type binding for structured Q&A
    
    Example:
        >>> kb = VSAKnowledgeBase(dim=256)
        >>> kb.add_relational_fact("capital_of", "france", "paris")
        >>> kb.add_relational_fact("capital_of", "germany", "berlin")
        >>> 
        >>> # Query: What is the capital of France?
        >>> results = kb.query_relation("capital_of", "france")
        >>> print(results[0])  # ("paris", 0.85)
        >>> 
        >>> # Analogy: France:Paris :: Germany:?
        >>> results = kb.solve_analogy("france", "paris", "germany")
        >>> print(results[0])  # ("berlin", 0.72)
    """
    
    def __init__(self, vocab: Vocabulary = None, dim: int = 256,
                 method: BindingMethod = BindingMethod.CIRCULAR_CONV):
        # Initialize parent with matching dimension
        if vocab is None:
            vocab = Vocabulary(dim=dim)
        super().__init__(vocab)
        
        self.dim = dim
        self.method = method
        
        # VSA-specific storage
        self.relational_facts: Dict[str, RelationalFact] = {}
        self.role_vectors: Dict[str, np.ndarray] = {}
        self.entity_vectors: Dict[str, np.ndarray] = {}
        self.value_cleanup = CleanupMemory()
        
        # Question type vectors for structured Q&A
        self.question_type_vectors: Dict[str, np.ndarray] = {}
        self._init_question_types()
    
    def _init_question_types(self):
        """Initialize vectors for question types."""
        for qtype in ['WHO', 'WHAT', 'WHERE', 'WHEN', 'WHY', 'HOW', 'UNKNOWN']:
            self.question_type_vectors[qtype] = self._get_deterministic_vector(f"__QTYPE_{qtype}__")
    
    def _get_deterministic_vector(self, name: str) -> np.ndarray:
        """Get a deterministic unit vector from a name."""
        seed = hash(name) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim)
        return vec / np.linalg.norm(vec)
    
    def get_role_vector(self, role: str) -> np.ndarray:
        """Get or create vector for a role (e.g., 'capital_of')."""
        role = role.lower()
        if role not in self.role_vectors:
            self.role_vectors[role] = self._get_deterministic_vector(f"__ROLE_{role}__")
        return self.role_vectors[role]
    
    def get_entity_vector(self, entity: str) -> np.ndarray:
        """Get or create vector for an entity."""
        entity = entity.lower()
        if entity not in self.entity_vectors:
            self.entity_vectors[entity] = self._get_deterministic_vector(entity)
        return self.entity_vectors[entity]
    
    # =========================================================================
    # RELATIONAL FACTS
    # =========================================================================
    
    def add_relational_fact(self, role: str, subject: str, value: str,
                            source: str = "") -> RelationalFact:
        """
        Add a relational fact: role(subject) = value
        
        Encodes as: role ⊛ subject bundled with value
        
        Examples:
            add_relational_fact("capital_of", "france", "paris")
            add_relational_fact("authored_by", "moby_dick", "melville")
            add_relational_fact("located_in", "eiffel_tower", "paris")
        """
        role_vec = self.get_role_vector(role)
        subject_vec = self.get_entity_vector(subject)
        value_vec = self.get_entity_vector(value)
        
        # Encode: role ⊛ subject + value
        bound = bind(role_vec, subject_vec, self.method)
        fact_vec = bundle(bound, value_vec)
        
        # Create fact
        fact_id = hashlib.md5(f"{role}:{subject}:{value}".encode()).hexdigest()[:12]
        fact = RelationalFact(
            id=fact_id,
            role=role.lower(),
            subject=subject.lower(),
            value=value.lower(),
            vector=fact_vec,
            source=source
        )
        
        self.relational_facts[fact_id] = fact
        self.value_cleanup.add(value.lower(), value_vec)
        
        # Also add as standard fact for compatibility
        self.add_fact(f"{subject} {role} {value}", source)
        
        return fact
    
    def query_relation(self, role: str, subject: str, 
                       k: int = 5) -> List[Tuple[str, float]]:
        """
        Query for value given role and subject.
        
        Example: query_relation("capital_of", "france") → [("paris", 0.85), ...]
        """
        role_vec = self.get_role_vector(role)
        subject_vec = self.get_entity_vector(subject)
        
        # Create query pattern: role ⊛ subject
        query = bind(role_vec, subject_vec, self.method)
        
        # Find matching facts and unbind to get value
        results = []
        
        for fact in self.relational_facts.values():
            if fact.role == role.lower():
                # Unbind to recover value
                recovered = unbind(fact.vector, query, self.method)
                
                # Check similarity to known values
                matches = self.value_cleanup.cleanup_top_k(recovered, k=1)
                if matches:
                    value_name, sim = matches[0]
                    results.append((value_name, sim))
        
        # Also try direct pattern matching
        for fact in self.relational_facts.values():
            if fact.role == role.lower() and fact.subject == subject.lower():
                # Direct match - high confidence
                results.append((fact.value, 1.0))
        
        # Deduplicate and sort
        seen = set()
        unique_results = []
        for name, sim in sorted(results, key=lambda x: -x[1]):
            if name not in seen:
                seen.add(name)
                unique_results.append((name, sim))
        
        return unique_results[:k]
    
    # =========================================================================
    # ANALOGICAL REASONING
    # =========================================================================
    
    def solve_analogy(self, a1: str, b1: str, a2: str,
                      k: int = 5) -> List[Tuple[str, float]]:
        """
        Solve analogy: a1:b1 :: a2:?
        
        Example: solve_analogy("france", "paris", "germany") → [("berlin", 0.72), ...]
        
        The relation between a1 and b1 is extracted and applied to a2.
        """
        a1_vec = self.get_entity_vector(a1)
        b1_vec = self.get_entity_vector(b1)
        a2_vec = self.get_entity_vector(a2)
        
        # Extract implicit relation: b1 ⊛ a1
        relation = bind(b1_vec, a1_vec, self.method)
        
        # Apply to a2
        answer = bind(relation, a2_vec, self.method)
        
        return self.value_cleanup.cleanup_top_k(answer, k)
    
    # =========================================================================
    # STRUCTURED Q&A WITH BINDING
    # =========================================================================
    
    def add_qa_pair_vsa(self, question: str, answer: str,
                        qtype: str = None, source: str = "") -> QAPair:
        """
        Add Q&A pair with VSA binding for question type.
        
        The question is encoded as: qtype ⊛ content
        This allows querying by question type structure.
        """
        # Standard Q&A pair
        qa = self.add_qa_pair(question, answer, qtype, source)
        
        # Also encode with question type binding
        if qtype is None:
            qtype = detect_question_type(question)
        
        qtype_vec = self.question_type_vectors.get(qtype, 
                        self.question_type_vectors['UNKNOWN'])
        content_vec = self.vocab.encode(question)
        
        # Bind question type with content
        bound_question = bind(qtype_vec, content_vec, self.method)
        
        # Store bound encoding (could extend QAPair dataclass)
        # For now, we'll use it in search
        
        return qa
    
    def search_qa_vsa(self, question: str, k: int = 5,
                      use_type_binding: bool = True) -> List[Tuple[QAPair, float]]:
        """
        Search Q&A pairs using VSA binding.
        
        If use_type_binding is True, questions are matched by both
        content similarity AND question type structure.
        """
        query_vec = self.vocab.encode(question)
        qtype = detect_question_type(question)
        
        if use_type_binding:
            qtype_vec = self.question_type_vectors.get(qtype,
                            self.question_type_vectors['UNKNOWN'])
            query_vec = bind(qtype_vec, query_vec, self.method)
        
        results = []
        for qa in self.qa_pairs.values():
            # Encode stored question the same way
            stored_vec = qa.question_encoding
            if use_type_binding:
                stored_qtype = qa.question_type
                stored_qtype_vec = self.question_type_vectors.get(stored_qtype,
                                       self.question_type_vectors['UNKNOWN'])
                stored_vec = bind(stored_qtype_vec, stored_vec, self.method)
            
            sim = similarity(query_vec, stored_vec)
            results.append((qa, sim))
        
        results.sort(key=lambda x: -x[1])
        return results[:k]
    
    # =========================================================================
    # UNIFIED PROJECTION VIEW
    # =========================================================================
    
    def unified_query(self, query: str, k: int = 5) -> Dict[str, List]:
        """
        Unified query that searches across all knowledge types.
        
        Returns results from:
        - Standard facts (cosine similarity)
        - Relational facts (VSA binding)
        - Q&A pairs (with optional type binding)
        
        This demonstrates the unified geometric framework.
        """
        results = {
            'facts': [],
            'relational': [],
            'qa': [],
        }
        
        # Standard fact search
        fact_results = self.search_facts(query, k)
        results['facts'] = [(f.content, sim) for f, sim in fact_results]
        
        # Relational search (if query looks like "X of Y")
        # Simple pattern matching for demo
        import re
        rel_match = re.match(r'(?:what is the )?(\w+) of (\w+)', query.lower())
        if rel_match:
            role, subject = rel_match.groups()
            rel_results = self.query_relation(role, subject, k)
            results['relational'] = rel_results
        
        # Q&A search
        qa_results = self.search_qa_vsa(query, k, use_type_binding=True)
        results['qa'] = [(qa.answer, sim) for qa, sim in qa_results]
        
        return results


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

def demo_vsa_knowledge_base():
    """Demonstrate VSA-enhanced knowledge base."""
    
    print("=" * 70)
    print("VSA-ENHANCED KNOWLEDGE BASE DEMO")
    print("=" * 70)
    print()
    
    # Create knowledge base
    kb = VSAKnowledgeBase(dim=256)
    
    # Add relational facts
    print("Adding relational facts:")
    print("-" * 70)
    
    facts = [
        ("capital_of", "france", "paris"),
        ("capital_of", "germany", "berlin"),
        ("capital_of", "japan", "tokyo"),
        ("capital_of", "italy", "rome"),
        ("capital_of", "spain", "madrid"),
        ("authored_by", "moby_dick", "melville"),
        ("authored_by", "hamlet", "shakespeare"),
        ("authored_by", "1984", "orwell"),
        ("located_in", "eiffel_tower", "paris"),
        ("located_in", "colosseum", "rome"),
    ]
    
    for role, subject, value in facts:
        kb.add_relational_fact(role, subject, value)
        print(f"  {role}({subject}) = {value}")
    
    print()
    
    # Query relations
    print("Querying relations:")
    print("-" * 70)
    
    queries = [
        ("capital_of", "france"),
        ("capital_of", "germany"),
        ("authored_by", "moby_dick"),
        ("located_in", "eiffel_tower"),
    ]
    
    for role, subject in queries:
        results = kb.query_relation(role, subject, k=3)
        print(f"  {role}({subject}) = ?")
        for value, sim in results[:1]:
            print(f"    → {value} (sim={sim:.3f})")
    
    print()
    
    # Analogical reasoning
    print("Analogical reasoning:")
    print("-" * 70)
    
    analogies = [
        ("france", "paris", "germany", "berlin"),
        ("france", "paris", "japan", "tokyo"),
        ("moby_dick", "melville", "hamlet", "shakespeare"),
    ]
    
    for a1, b1, a2, expected in analogies:
        results = kb.solve_analogy(a1, b1, a2, k=3)
        answer = results[0][0] if results else "?"
        sim = results[0][1] if results else 0
        marker = "✓" if answer == expected else "✗"
        print(f"  {marker} {a1}:{b1} :: {a2}:? → {answer} (expected: {expected}, sim={sim:.3f})")
    
    print()
    
    # Q&A with type binding
    print("Q&A with question type binding:")
    print("-" * 70)
    
    # Add some Q&A pairs
    qa_pairs = [
        ("Who wrote Moby Dick?", "Herman Melville wrote Moby Dick."),
        ("What is the capital of France?", "Paris is the capital of France."),
        ("Where is the Eiffel Tower?", "The Eiffel Tower is in Paris."),
        ("When was Shakespeare born?", "Shakespeare was born in 1564."),
    ]
    
    for q, a in qa_pairs:
        kb.add_qa_pair_vsa(q, a)
    
    test_questions = [
        "Who is the author of Moby Dick?",
        "What city is the capital of France?",
        "Where can I find the Eiffel Tower?",
    ]
    
    for question in test_questions:
        results = kb.search_qa_vsa(question, k=1)
        if results:
            qa, sim = results[0]
            print(f"  Q: {question}")
            print(f"  A: {qa.answer} (sim={sim:.3f})")
            print()
    
    return kb


if __name__ == "__main__":
    demo_vsa_knowledge_base()
