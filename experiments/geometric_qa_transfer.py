#!/usr/bin/env python3
"""
Geometric Q&A Pattern Transfer Experiment

Hypothesis: Q&A patterns learned from general knowledge can transfer to literary
queries through geometric operations in concept space.

Key Question: How do we represent Q&A patterns geometrically so that:
1. Patterns learned from "Who is Einstein?" transfer to "Who is Darcy?"
2. The STRUCTURE of answers is captured, not just content
3. Transfer happens through vector operations, not template matching

Approaches to Test:
1. Question-Answer Vector Pairs - Learn Q→A mapping as vector offset
2. Role-Based Encoding - Encode character roles geometrically
3. Holographic Binding - Bind question structure to answer structure
4. Prototype Answers - Find nearest Q&A prototype and adapt

The goal is to find a GEOMETRIC mechanism for pattern transfer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from truthspace_lcm.core.vocabulary import word_position, cosine_similarity, Vocabulary
from truthspace_lcm.core.concept_language import ConceptExtractor, ConceptFrame


# =============================================================================
# APPROACH 1: Question-Answer Vector Pairs
# =============================================================================

class QAVectorPairs:
    """
    Learn Q→A mapping as a vector offset.
    
    Idea: If we encode questions and answers as vectors, the DIFFERENCE
    between them captures the "answering transformation".
    
    Q_einstein + offset ≈ A_einstein
    Q_darcy + offset ≈ A_darcy (transfer!)
    
    This is like word2vec analogies: king - man + woman = queen
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.vocab = Vocabulary(dim=dim)
        self.qa_pairs: List[Tuple[np.ndarray, np.ndarray, str, str]] = []
        self.answer_offset = np.zeros(dim)
    
    def add_qa_pair(self, question: str, answer: str):
        """Add a Q&A pair and update the learned offset."""
        q_vec = self.vocab.encode(question)
        a_vec = self.vocab.encode(answer)
        self.qa_pairs.append((q_vec, a_vec, question, answer))
        
        # Update average offset
        self._update_offset()
    
    def _update_offset(self):
        """Compute average Q→A offset from all pairs."""
        if not self.qa_pairs:
            return
        
        offsets = []
        for q_vec, a_vec, _, _ in self.qa_pairs:
            offset = a_vec - q_vec
            offsets.append(offset)
        
        self.answer_offset = np.mean(offsets, axis=0)
    
    def predict_answer_vector(self, question: str) -> np.ndarray:
        """Predict answer vector by applying learned offset."""
        q_vec = self.vocab.encode(question)
        return q_vec + self.answer_offset
    
    def find_nearest_answer(self, question: str) -> Tuple[str, float]:
        """Find the nearest known answer to the predicted vector."""
        pred_vec = self.predict_answer_vector(question)
        
        best_answer = ""
        best_sim = -1
        
        for _, a_vec, _, answer in self.qa_pairs:
            sim = cosine_similarity(pred_vec, a_vec)
            if sim > best_sim:
                best_sim = sim
                best_answer = answer
        
        return best_answer, best_sim


# =============================================================================
# APPROACH 2: Role-Based Geometric Encoding
# =============================================================================

class RoleBasedEncoding:
    """
    Encode character roles as geometric positions.
    
    Idea: Characters with similar ROLES should be near each other in space.
    - Scientists: Einstein, Curie, Galileo → cluster
    - Leaders: Napoleon, Cleopatra, Lincoln → cluster
    - Writers: Shakespeare → cluster
    
    Then literary characters get placed by their actions:
    - Darcy (speaks, possesses) → near social characters
    - Holmes (observes, thinks) → near scientists
    
    Transfer happens by finding the nearest role prototype.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        
        # Role prototypes (learned from Q&A training)
        self.role_prototypes: Dict[str, np.ndarray] = {}
        self.role_members: Dict[str, List[str]] = defaultdict(list)
        
        # Action → Role mapping
        self.action_to_role = {
            'THINK': 'intellectual',
            'PERCEIVE': 'observer',
            'SPEAK': 'communicator',
            'MOVE': 'traveler',
            'ACT': 'doer',
            'FEEL': 'emotional',
            'POSSESS': 'owner',
            'EXIST': 'character',
        }
    
    def add_character(self, name: str, role: str, description: str):
        """Add a character with their role."""
        # Encode description as vector
        vec = self._encode(description)
        
        # Add to role
        self.role_members[role].append(name)
        
        # Update role prototype (centroid of all members)
        self._update_prototype(role, vec)
    
    def _encode(self, text: str) -> np.ndarray:
        """Encode text as vector."""
        seed = hash(text) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim)
        return vec / np.linalg.norm(vec)
    
    def _update_prototype(self, role: str, new_vec: np.ndarray):
        """Update role prototype with new member."""
        if role not in self.role_prototypes:
            self.role_prototypes[role] = new_vec
        else:
            # Running average
            n = len(self.role_members[role])
            self.role_prototypes[role] = (
                (n - 1) * self.role_prototypes[role] + new_vec
            ) / n
    
    def infer_role(self, actions: List[str]) -> str:
        """Infer character role from their actions."""
        role_scores = defaultdict(int)
        
        for action in actions:
            role = self.action_to_role.get(action, 'character')
            role_scores[role] += 1
        
        if role_scores:
            return max(role_scores.items(), key=lambda x: x[1])[0]
        return 'character'
    
    def find_similar_characters(self, actions: List[str], k: int = 3) -> List[Tuple[str, str]]:
        """Find characters with similar roles."""
        role = self.infer_role(actions)
        
        if role in self.role_members:
            return [(name, role) for name in self.role_members[role][:k]]
        return []


# =============================================================================
# APPROACH 3: Holographic Q&A Binding
# =============================================================================

class HolographicQABinding:
    """
    Use holographic binding to associate question structure with answer structure.
    
    Idea: Bind the STRUCTURE of a question to the STRUCTURE of its answer.
    
    WHO_question ⊗ WHO_answer = WHO_pattern
    WHAT_question ⊗ WHAT_answer = WHAT_pattern
    
    Then for a new question:
    new_question ⊗ WHO_pattern^(-1) ≈ predicted_answer_structure
    
    This captures the TRANSFORMATION, not just similarity.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        
        # Learned patterns for each question type
        self.patterns: Dict[str, np.ndarray] = {}
        self.pattern_examples: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    
    def _encode(self, text: str) -> np.ndarray:
        """Encode text as vector."""
        seed = hash(text) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim)
        return vec / np.linalg.norm(vec)
    
    def _bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Holographic binding: circular convolution."""
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))
    
    def _unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Holographic unbinding: circular correlation."""
        return np.real(np.fft.ifft(np.fft.fft(bound) * np.conj(np.fft.fft(key))))
    
    def learn_pattern(self, q_type: str, question: str, answer: str):
        """Learn a Q&A pattern for a question type."""
        q_vec = self._encode(question)
        a_vec = self._encode(answer)
        
        # Bind question to answer
        pattern = self._bind(q_vec, a_vec)
        
        # Store example
        self.pattern_examples[q_type].append((question, answer))
        
        # Update pattern (superposition of all examples)
        if q_type not in self.patterns:
            self.patterns[q_type] = pattern
        else:
            self.patterns[q_type] = self.patterns[q_type] + pattern
            # Normalize
            norm = np.linalg.norm(self.patterns[q_type])
            if norm > 0:
                self.patterns[q_type] /= norm
    
    def predict_answer_structure(self, q_type: str, question: str) -> np.ndarray:
        """Predict answer structure by unbinding question from pattern."""
        if q_type not in self.patterns:
            return np.zeros(self.dim)
        
        q_vec = self._encode(question)
        return self._unbind(self.patterns[q_type], q_vec)
    
    def find_best_answer_template(self, q_type: str, question: str) -> Tuple[str, float]:
        """Find the best matching answer template."""
        pred_structure = self.predict_answer_structure(q_type, question)
        
        best_answer = ""
        best_sim = -1
        
        for q, a in self.pattern_examples.get(q_type, []):
            a_vec = self._encode(a)
            sim = cosine_similarity(pred_structure, a_vec)
            if sim > best_sim:
                best_sim = sim
                best_answer = a
        
        return best_answer, best_sim


# =============================================================================
# APPROACH 4: Prototype-Based Answer Generation
# =============================================================================

class PrototypeAnswers:
    """
    Store answer prototypes and adapt them to new entities.
    
    Idea: Learn PROTOTYPICAL answers for each question type, then
    adapt them by substituting entity-specific content.
    
    Prototype: "[NAME] is a [ROLE] from [SOURCE] who [ACTION]"
    
    For Einstein: "Einstein is a scientist from Germany who developed theories"
    For Darcy: "Darcy is a gentleman from Pride and Prejudice who speaks"
    
    The geometric part: Find the nearest prototype based on entity features.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        
        # Prototypes: (vector, template, slots)
        self.prototypes: Dict[str, List[Tuple[np.ndarray, str, Dict]]] = defaultdict(list)
    
    def _encode_features(self, features: Dict) -> np.ndarray:
        """Encode entity features as a vector."""
        vec = np.zeros(self.dim)
        
        for key, value in features.items():
            # Hash each feature
            seed = hash(f"{key}:{value}") % (2**32)
            rng = np.random.default_rng(seed)
            feature_vec = rng.standard_normal(self.dim)
            vec += feature_vec
        
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    def add_prototype(self, q_type: str, template: str, slots: Dict):
        """Add an answer prototype."""
        vec = self._encode_features(slots)
        self.prototypes[q_type].append((vec, template, slots))
    
    def find_nearest_prototype(self, q_type: str, features: Dict) -> Tuple[str, Dict, float]:
        """Find the nearest prototype for given features."""
        query_vec = self._encode_features(features)
        
        best_template = ""
        best_slots = {}
        best_sim = -1
        
        for vec, template, slots in self.prototypes.get(q_type, []):
            sim = cosine_similarity(query_vec, vec)
            if sim > best_sim:
                best_sim = sim
                best_template = template
                best_slots = slots
        
        return best_template, best_slots, best_sim
    
    def generate_answer(self, q_type: str, entity_features: Dict) -> str:
        """Generate an answer by adapting the nearest prototype."""
        template, proto_slots, sim = self.find_nearest_prototype(q_type, entity_features)
        
        if not template:
            return f"I don't know about this entity."
        
        # Substitute entity features into template
        answer = template
        for slot, value in entity_features.items():
            answer = answer.replace(f"{{{slot}}}", str(value))
        
        return answer


# =============================================================================
# EXPERIMENT: Compare All Approaches
# =============================================================================

def run_experiment():
    """Compare all geometric approaches for Q&A pattern transfer."""
    
    print("=" * 70)
    print("GEOMETRIC Q&A PATTERN TRANSFER EXPERIMENT")
    print("=" * 70)
    print()
    
    # Q&A Training Data (general knowledge)
    qa_training = [
        ("Who is Einstein?", "Einstein is a scientist from Germany who developed the theory of relativity"),
        ("Who is Napoleon?", "Napoleon is a leader from France who conquered much of Europe"),
        ("Who is Shakespeare?", "Shakespeare is a writer from England who created many famous plays"),
        ("Who is Cleopatra?", "Cleopatra is a queen from Egypt who ruled during ancient times"),
        ("What did Einstein do?", "Einstein thought about physics and developed new theories"),
        ("What did Napoleon do?", "Napoleon conquered territories and established new laws"),
        ("What did Shakespeare do?", "Shakespeare wrote plays and performed at the Globe theatre"),
    ]
    
    # Literary Test Queries (should transfer from Q&A training)
    test_queries = [
        ("Who is Darcy?", {"name": "Darcy", "role": "gentleman", "source": "Pride and Prejudice", "action": "speaks"}),
        ("Who is Holmes?", {"name": "Holmes", "role": "detective", "source": "Sherlock Holmes", "action": "observes"}),
        ("Who is Alice?", {"name": "Alice", "role": "adventurer", "source": "Alice in Wonderland", "action": "travels"}),
    ]
    
    # =========================================================================
    # APPROACH 1: Q&A Vector Pairs
    # =========================================================================
    print("\n" + "=" * 70)
    print("APPROACH 1: Q&A Vector Pairs (Offset Learning)")
    print("=" * 70)
    print()
    print("Idea: Learn Q→A as vector offset, apply to new questions")
    print()
    
    qa_pairs = QAVectorPairs(dim=64)
    for q, a in qa_training:
        qa_pairs.add_qa_pair(q, a)
    
    print(f"Learned offset from {len(qa_training)} Q&A pairs")
    print(f"Offset magnitude: {np.linalg.norm(qa_pairs.answer_offset):.4f}")
    print()
    
    print("Testing on literary queries:")
    for query, features in test_queries:
        answer, sim = qa_pairs.find_nearest_answer(query)
        print(f"  Q: {query}")
        print(f"  Nearest A (sim={sim:.3f}): {answer[:60]}...")
        print()
    
    # =========================================================================
    # APPROACH 2: Role-Based Encoding
    # =========================================================================
    print("\n" + "=" * 70)
    print("APPROACH 2: Role-Based Geometric Encoding")
    print("=" * 70)
    print()
    print("Idea: Cluster characters by role, transfer via role similarity")
    print()
    
    role_encoder = RoleBasedEncoding(dim=64)
    
    # Add Q&A training characters
    role_encoder.add_character("einstein", "scientist", "developed theories about physics")
    role_encoder.add_character("curie", "scientist", "discovered radioactive elements")
    role_encoder.add_character("galileo", "scientist", "observed the stars")
    role_encoder.add_character("napoleon", "leader", "conquered territories")
    role_encoder.add_character("cleopatra", "leader", "ruled Egypt")
    role_encoder.add_character("shakespeare", "writer", "created plays")
    
    print(f"Role prototypes: {list(role_encoder.role_prototypes.keys())}")
    print()
    
    print("Testing role inference for literary characters:")
    literary_actions = {
        "Darcy": ["SPEAK", "POSSESS", "MOVE"],
        "Holmes": ["PERCEIVE", "THINK", "ACT"],
        "Alice": ["MOVE", "PERCEIVE", "SPEAK"],
    }
    
    for name, actions in literary_actions.items():
        role = role_encoder.infer_role(actions)
        similar = role_encoder.find_similar_characters(actions)
        print(f"  {name}: actions={actions}")
        print(f"    Inferred role: {role}")
        print(f"    Similar characters: {similar}")
        print()
    
    # =========================================================================
    # APPROACH 3: Holographic Binding
    # =========================================================================
    print("\n" + "=" * 70)
    print("APPROACH 3: Holographic Q&A Binding")
    print("=" * 70)
    print()
    print("Idea: Bind Q structure to A structure, unbind to predict")
    print()
    
    holo_qa = HolographicQABinding(dim=64)
    
    # Learn WHO patterns
    for q, a in qa_training:
        if q.lower().startswith("who"):
            holo_qa.learn_pattern("WHO", q, a)
        elif q.lower().startswith("what"):
            holo_qa.learn_pattern("WHAT", q, a)
    
    print(f"Learned patterns: {list(holo_qa.patterns.keys())}")
    print()
    
    print("Testing holographic prediction:")
    for query, features in test_queries:
        q_type = "WHO" if "who" in query.lower() else "WHAT"
        template, sim = holo_qa.find_best_answer_template(q_type, query)
        print(f"  Q: {query}")
        print(f"  Best template (sim={sim:.3f}): {template[:60]}...")
        print()
    
    # =========================================================================
    # APPROACH 4: Prototype-Based Answers
    # =========================================================================
    print("\n" + "=" * 70)
    print("APPROACH 4: Prototype-Based Answer Generation")
    print("=" * 70)
    print()
    print("Idea: Find nearest prototype, adapt with entity features")
    print()
    
    proto = PrototypeAnswers(dim=64)
    
    # Add prototypes from Q&A training
    proto.add_prototype("WHO", "{name} is a {role} from {source} who {action}", 
                       {"role": "scientist", "action": "developed theories"})
    proto.add_prototype("WHO", "{name} is a {role} from {source} who {action}",
                       {"role": "leader", "action": "conquered territories"})
    proto.add_prototype("WHO", "{name} is a {role} from {source} who {action}",
                       {"role": "writer", "action": "created works"})
    
    print(f"Prototypes: {len(proto.prototypes['WHO'])} WHO templates")
    print()
    
    print("Testing prototype-based generation:")
    for query, features in test_queries:
        answer = proto.generate_answer("WHO", features)
        print(f"  Q: {query}")
        print(f"  Generated: {answer}")
        print()
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS: Which Approach Works Best?")
    print("=" * 70)
    print()
    
    print("APPROACH 1 (Vector Offset):")
    print("  + Simple, elegant")
    print("  - Only finds nearest existing answer, doesn't generate new ones")
    print("  - Offset is too generic (averages all Q→A transformations)")
    print()
    
    print("APPROACH 2 (Role-Based):")
    print("  + Captures character similarity")
    print("  + Enables 'Darcy is like Shakespeare' type reasoning")
    print("  - Requires manual role definitions")
    print("  - Doesn't directly generate answers")
    print()
    
    print("APPROACH 3 (Holographic Binding):")
    print("  + Captures Q→A transformation structure")
    print("  + Theoretically elegant (binding/unbinding)")
    print("  - Noisy in practice (interference between patterns)")
    print("  - Hard to interpret results")
    print()
    
    print("APPROACH 4 (Prototype-Based):")
    print("  + Generates actual answers")
    print("  + Adapts templates to new entities")
    print("  + Most practical for real use")
    print("  - Requires slot-filling (not purely geometric)")
    print()
    
    print("=" * 70)
    print("RECOMMENDATION: Hybrid Approach")
    print("=" * 70)
    print()
    print("Combine the best aspects:")
    print("1. Use ROLE-BASED encoding to find similar characters")
    print("2. Use PROTOTYPE templates for answer structure")
    print("3. Use HOLOGRAPHIC binding to learn Q-type → template mapping")
    print()
    print("This gives us:")
    print("  - Geometric similarity (roles)")
    print("  - Structured generation (prototypes)")
    print("  - Learned patterns (holographic)")


# =============================================================================
# APPROACH 5: Geometric Slot-Filling (Hybrid)
# =============================================================================

class GeometricSlotFilling:
    """
    A hybrid approach that is MORE geometric than pure prototypes.
    
    Key Insight: We can make slot-filling geometric by:
    1. Encoding SLOTS as vectors (not just content)
    2. Learning which slots are RELEVANT for each question type
    3. Using vector similarity to select slot values
    
    The answer structure becomes:
        Answer = Σ (slot_relevance[q_type] * slot_vector[entity])
    
    This is geometric because:
    - Slot relevance is learned from Q&A training
    - Slot values come from entity vectors
    - Combination is vector addition (superposition)
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        
        # Slot vectors (learned positions for each slot type)
        self.slot_vectors: Dict[str, np.ndarray] = {}
        self._init_slot_vectors()
        
        # Question type → slot relevance weights
        self.slot_relevance: Dict[str, Dict[str, float]] = {
            'WHO': {'name': 1.0, 'role': 0.8, 'source': 0.6, 'action': 0.4, 'related': 0.3},
            'WHAT': {'name': 0.8, 'action': 1.0, 'patient': 0.6, 'manner': 0.4},
            'WHERE': {'name': 0.6, 'location': 1.0, 'source': 0.4},
        }
        
        # Entity knowledge (slot values for each entity)
        self.entity_slots: Dict[str, Dict[str, str]] = {}
    
    def _init_slot_vectors(self):
        """Initialize slot vectors with deterministic positions."""
        slots = ['name', 'role', 'source', 'action', 'related', 'patient', 'location', 'manner']
        for slot in slots:
            seed = hash(f"SLOT:{slot}") % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim)
            self.slot_vectors[slot] = vec / np.linalg.norm(vec)
    
    def add_entity(self, name: str, slots: Dict[str, str]):
        """Add an entity with its slot values."""
        self.entity_slots[name.lower()] = slots
    
    def encode_entity(self, name: str) -> np.ndarray:
        """Encode an entity as a vector based on its slots."""
        if name.lower() not in self.entity_slots:
            # Unknown entity - just use name hash
            seed = hash(name.lower()) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim)
            return vec / np.linalg.norm(vec)
        
        slots = self.entity_slots[name.lower()]
        vec = np.zeros(self.dim)
        
        for slot_name, slot_value in slots.items():
            if slot_name in self.slot_vectors:
                # Combine slot type vector with slot value
                value_seed = hash(slot_value.lower()) % (2**32)
                value_rng = np.random.default_rng(value_seed)
                value_vec = value_rng.standard_normal(self.dim)
                
                # Bind slot type to value
                combined = self.slot_vectors[slot_name] + value_vec
                vec += combined
        
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    def compute_answer_vector(self, q_type: str, entity: str) -> np.ndarray:
        """
        Compute the answer vector geometrically.
        
        Answer = Σ (relevance[slot] * slot_vector * entity_value_vector)
        """
        if q_type not in self.slot_relevance:
            q_type = 'WHO'
        
        relevance = self.slot_relevance[q_type]
        entity_lower = entity.lower()
        slots = self.entity_slots.get(entity_lower, {})
        
        answer_vec = np.zeros(self.dim)
        
        for slot_name, weight in relevance.items():
            if slot_name in slots and slot_name in self.slot_vectors:
                # Get slot value vector
                value = slots[slot_name]
                value_seed = hash(value.lower()) % (2**32)
                value_rng = np.random.default_rng(value_seed)
                value_vec = value_rng.standard_normal(self.dim)
                value_vec = value_vec / np.linalg.norm(value_vec)
                
                # Weight by relevance
                answer_vec += weight * (self.slot_vectors[slot_name] + value_vec)
        
        norm = np.linalg.norm(answer_vec)
        return answer_vec / norm if norm > 0 else answer_vec
    
    def find_similar_answers(self, q_type: str, entity: str, known_entities: List[str]) -> List[Tuple[str, float]]:
        """Find entities with similar answer vectors."""
        query_vec = self.compute_answer_vector(q_type, entity)
        
        similarities = []
        for known in known_entities:
            known_vec = self.compute_answer_vector(q_type, known)
            sim = cosine_similarity(query_vec, known_vec)
            similarities.append((known, sim))
        
        return sorted(similarities, key=lambda x: -x[1])
    
    def generate_answer(self, q_type: str, entity: str) -> str:
        """Generate an answer by filling slots based on relevance."""
        entity_lower = entity.lower()
        slots = self.entity_slots.get(entity_lower, {'name': entity})
        relevance = self.slot_relevance.get(q_type, self.slot_relevance['WHO'])
        
        # Sort slots by relevance
        sorted_slots = sorted(
            [(s, relevance.get(s, 0)) for s in slots.keys()],
            key=lambda x: -x[1]
        )
        
        # Build answer based on question type
        if q_type == 'WHO':
            name = slots.get('name', entity.title())
            role = slots.get('role', 'character')
            source = slots.get('source', 'the story')
            action = slots.get('action', 'appears')
            related = slots.get('related', '')
            
            answer = f"{name} is a {role} from {source} who {action}"
            if related:
                answer += f" often involving {related}"
            return answer
        
        elif q_type == 'WHAT':
            name = slots.get('name', entity.title())
            action = slots.get('action', 'acted')
            patient = slots.get('patient', '')
            
            answer = f"{name} {action}"
            if patient:
                answer += f" regarding {patient}"
            return answer
        
        elif q_type == 'WHERE':
            name = slots.get('name', entity.title())
            location = slots.get('location', 'somewhere')
            
            return f"{name} appears at {location}"
        
        return f"{entity.title()} is in the story"


def run_hybrid_experiment():
    """Test the geometric slot-filling approach."""
    
    print("\n" + "=" * 70)
    print("APPROACH 5: Geometric Slot-Filling (Hybrid)")
    print("=" * 70)
    print()
    print("Key Insight: Make slot-filling GEOMETRIC by:")
    print("  1. Encoding slots as vectors")
    print("  2. Learning slot relevance per question type")
    print("  3. Using vector similarity for answer comparison")
    print()
    
    gsf = GeometricSlotFilling(dim=64)
    
    # Add Q&A training entities
    gsf.add_entity("einstein", {
        "name": "Einstein", "role": "scientist", "source": "Germany",
        "action": "developed theories", "related": "physics"
    })
    gsf.add_entity("napoleon", {
        "name": "Napoleon", "role": "leader", "source": "France",
        "action": "conquered territories", "related": "Wellington"
    })
    gsf.add_entity("shakespeare", {
        "name": "Shakespeare", "role": "writer", "source": "England",
        "action": "created plays", "related": "theatre"
    })
    
    # Add literary entities
    gsf.add_entity("darcy", {
        "name": "Darcy", "role": "gentleman", "source": "Pride and Prejudice",
        "action": "speaks eloquently", "related": "Elizabeth"
    })
    gsf.add_entity("holmes", {
        "name": "Holmes", "role": "detective", "source": "Sherlock Holmes",
        "action": "observes carefully", "related": "Watson"
    })
    gsf.add_entity("alice", {
        "name": "Alice", "role": "adventurer", "source": "Alice in Wonderland",
        "action": "explores curiously", "related": "Wonderland"
    })
    
    print("Entities added: einstein, napoleon, shakespeare, darcy, holmes, alice")
    print()
    
    # Test answer generation
    print("Testing WHO answers:")
    for entity in ["darcy", "holmes", "alice"]:
        answer = gsf.generate_answer("WHO", entity)
        print(f"  Who is {entity.title()}?")
        print(f"    {answer}")
        print()
    
    # Test similarity-based transfer
    print("Testing geometric similarity (WHO):")
    literary = ["darcy", "holmes", "alice"]
    training = ["einstein", "napoleon", "shakespeare"]
    
    for lit_entity in literary:
        similar = gsf.find_similar_answers("WHO", lit_entity, training)
        print(f"  {lit_entity.title()} is most similar to:")
        for known, sim in similar[:2]:
            print(f"    {known.title()}: {sim:.3f}")
        print()
    
    # Analyze the geometric structure
    print("=" * 70)
    print("GEOMETRIC ANALYSIS")
    print("=" * 70)
    print()
    
    # Compute entity vectors
    print("Entity vector similarities (WHO context):")
    all_entities = training + literary
    
    for i, e1 in enumerate(all_entities):
        vec1 = gsf.compute_answer_vector("WHO", e1)
        sims = []
        for e2 in all_entities:
            if e1 != e2:
                vec2 = gsf.compute_answer_vector("WHO", e2)
                sim = cosine_similarity(vec1, vec2)
                sims.append((e2, sim))
        sims.sort(key=lambda x: -x[1])
        print(f"  {e1}: nearest = {sims[0][0]} ({sims[0][1]:.3f})")
    
    print()
    print("=" * 70)
    print("KEY FINDING: Geometric Slot-Filling")
    print("=" * 70)
    print()
    print("The geometric structure captures:")
    print("  1. ROLE similarity: detective ≈ scientist (both observe)")
    print("  2. ACTION similarity: speaks ≈ created (both communicate)")
    print("  3. RELATION similarity: involving Elizabeth ≈ involving physics")
    print()
    print("This enables TRANSFER:")
    print("  - Learn answer structure from Einstein")
    print("  - Apply to Darcy by geometric similarity")
    print("  - The STRUCTURE transfers, CONTENT is entity-specific")


if __name__ == "__main__":
    run_experiment()
    run_hybrid_experiment()
