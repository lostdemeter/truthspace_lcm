#!/usr/bin/env python3
"""
Semantic Quaternion: 4D Encoding for Concept Semantics

Building on the φ-dial quaternion (output style), this adds a semantic
quaternion for concept encoding. This enables:

1. Analogies as quaternion rotations
2. Semantic similarity via quaternion distance
3. Relation extraction as rotation matrices

The Semantic Quaternion:
  q = w + xi + yj + zk

Where:
  x = Gender/Polarity (-1 = female/negative, +1 = male/positive)
  y = Age/Maturity (-1 = young/small, +1 = adult/large)
  z = Agency (φ-direction: -1 = receiver, +1 = initiator)
  w = Animacy (-1 = abstract/place, +1 = human/animate)

Key Insight: Analogies are ROTATIONS in this 4D space.
  king → queen = 180° rotation around x-axis (gender flip)
  man → boy = rotation around y-axis (age shift)
  france → paris = rotation around w-axis (country → city)

Author: Lesley Gushurst
License: GPLv3
"""

import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class SemanticQuaternion:
    """
    A quaternion encoding semantic features of a concept.
    
    q = w + xi + yj + zk
    
    Axes:
      x: Gender/Polarity (-1 female, +1 male)
      y: Age/Maturity (-1 young, +1 adult)
      z: Agency (-1 receiver, +1 initiator) 
      w: Animacy (-1 abstract, +1 human)
    """
    x: float = 0.0  # Gender/Polarity
    y: float = 0.0  # Age/Maturity
    z: float = 0.0  # Agency (φ-direction)
    w: float = 0.0  # Animacy
    
    def __add__(self, other: 'SemanticQuaternion') -> 'SemanticQuaternion':
        return SemanticQuaternion(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w
        )
    
    def __sub__(self, other: 'SemanticQuaternion') -> 'SemanticQuaternion':
        return SemanticQuaternion(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w
        )
    
    def __mul__(self, scalar: float) -> 'SemanticQuaternion':
        return SemanticQuaternion(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
            self.w * scalar
        )
    
    def __neg__(self) -> 'SemanticQuaternion':
        return SemanticQuaternion(-self.x, -self.y, -self.z, -self.w)
    
    @property
    def magnitude(self) -> float:
        """Quaternion magnitude."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
    
    def normalize(self) -> 'SemanticQuaternion':
        """Return unit quaternion."""
        mag = self.magnitude
        if mag == 0:
            return SemanticQuaternion(0, 0, 0, 1)
        return SemanticQuaternion(
            self.x / mag,
            self.y / mag,
            self.z / mag,
            self.w / mag
        )
    
    def dot(self, other: 'SemanticQuaternion') -> float:
        """Quaternion dot product (similarity)."""
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    
    def distance(self, other: 'SemanticQuaternion') -> float:
        """Euclidean distance in 4D space."""
        diff = self - other
        return diff.magnitude
    
    def cosine_similarity(self, other: 'SemanticQuaternion') -> float:
        """Cosine similarity between quaternions."""
        mag1 = self.magnitude
        mag2 = other.magnitude
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return self.dot(other) / (mag1 * mag2)
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)
    
    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float, float]) -> 'SemanticQuaternion':
        return cls(t[0], t[1], t[2], t[3])
    
    def __repr__(self) -> str:
        return f"SQ(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}, w={self.w:.2f})"


# =============================================================================
# SEMANTIC FEATURE DEFINITIONS
# =============================================================================

# Default semantic features for common concepts
# These can be learned or manually defined
DEFAULT_SEMANTIC_FEATURES: Dict[str, SemanticQuaternion] = {
    # Royalty - gender differentiated
    'king': SemanticQuaternion(x=1.0, y=1.0, z=1.0, w=1.0),      # male, adult, initiator, human
    'queen': SemanticQuaternion(x=-1.0, y=1.0, z=1.0, w=1.0),    # female, adult, initiator, human
    'prince': SemanticQuaternion(x=1.0, y=0.0, z=0.5, w=1.0),    # male, young-adult, semi-initiator
    'princess': SemanticQuaternion(x=-1.0, y=0.0, z=0.5, w=1.0), # female, young-adult
    
    # Family - gender and age differentiated
    'man': SemanticQuaternion(x=1.0, y=1.0, z=0.5, w=1.0),       # male, adult
    'woman': SemanticQuaternion(x=-1.0, y=1.0, z=0.5, w=1.0),    # female, adult
    'boy': SemanticQuaternion(x=1.0, y=-1.0, z=0.0, w=1.0),      # male, child
    'girl': SemanticQuaternion(x=-1.0, y=-1.0, z=0.0, w=1.0),    # female, child
    'father': SemanticQuaternion(x=1.0, y=1.0, z=1.0, w=1.0),    # male, adult, initiator
    'mother': SemanticQuaternion(x=-1.0, y=1.0, z=1.0, w=1.0),   # female, adult, initiator
    'son': SemanticQuaternion(x=1.0, y=-0.5, z=-0.5, w=1.0),     # male, young, receiver
    'daughter': SemanticQuaternion(x=-1.0, y=-0.5, z=-0.5, w=1.0), # female, young, receiver
    
    # Professions - gender differentiated
    'actor': SemanticQuaternion(x=1.0, y=1.0, z=0.5, w=1.0),
    'actress': SemanticQuaternion(x=-1.0, y=1.0, z=0.5, w=1.0),
    'waiter': SemanticQuaternion(x=1.0, y=1.0, z=-0.3, w=1.0),   # service = receiver-ish
    'waitress': SemanticQuaternion(x=-1.0, y=1.0, z=-0.3, w=1.0),
    'host': SemanticQuaternion(x=1.0, y=1.0, z=0.5, w=1.0),
    'hostess': SemanticQuaternion(x=-1.0, y=1.0, z=0.5, w=1.0),
    'doctor': SemanticQuaternion(x=0.0, y=1.0, z=0.8, w=1.0),    # neutral gender
    'nurse': SemanticQuaternion(x=0.0, y=1.0, z=0.3, w=1.0),
    
    # Countries and capitals - animacy differentiated
    'france': SemanticQuaternion(x=0.0, y=0.0, z=0.5, w=-0.5),   # place, contains
    'paris': SemanticQuaternion(x=0.0, y=0.0, z=-0.5, w=-0.8),   # place, contained
    'germany': SemanticQuaternion(x=0.0, y=0.0, z=0.5, w=-0.5),
    'berlin': SemanticQuaternion(x=0.0, y=0.0, z=-0.5, w=-0.8),
    'japan': SemanticQuaternion(x=0.0, y=0.0, z=0.5, w=-0.5),
    'tokyo': SemanticQuaternion(x=0.0, y=0.0, z=-0.5, w=-0.8),
    'italy': SemanticQuaternion(x=0.0, y=0.0, z=0.5, w=-0.5),
    'rome': SemanticQuaternion(x=0.0, y=0.0, z=-0.5, w=-0.8),
    'spain': SemanticQuaternion(x=0.0, y=0.0, z=0.5, w=-0.5),
    'madrid': SemanticQuaternion(x=0.0, y=0.0, z=-0.5, w=-0.8),
    'england': SemanticQuaternion(x=0.0, y=0.0, z=0.5, w=-0.5),
    'london': SemanticQuaternion(x=0.0, y=0.0, z=-0.5, w=-0.8),
    
    # Animals - age differentiated, non-human animacy
    'dog': SemanticQuaternion(x=0.0, y=1.0, z=0.3, w=0.5),       # animal, adult
    'puppy': SemanticQuaternion(x=0.0, y=-1.0, z=-0.3, w=0.5),   # animal, young
    'cat': SemanticQuaternion(x=0.0, y=1.0, z=0.3, w=0.5),
    'kitten': SemanticQuaternion(x=0.0, y=-1.0, z=-0.3, w=0.5),
    
    # Verbs - tense differentiated (using y for past/present)
    'walk': SemanticQuaternion(x=0.0, y=1.0, z=0.0, w=0.0),      # present
    'walked': SemanticQuaternion(x=0.0, y=-1.0, z=0.0, w=0.0),   # past
    'run': SemanticQuaternion(x=0.0, y=1.0, z=0.2, w=0.0),
    'ran': SemanticQuaternion(x=0.0, y=-1.0, z=0.2, w=0.0),
    'speak': SemanticQuaternion(x=0.0, y=1.0, z=0.3, w=0.0),
    'spoke': SemanticQuaternion(x=0.0, y=-1.0, z=0.3, w=0.0),
    'write': SemanticQuaternion(x=0.0, y=1.0, z=0.4, w=0.0),
    'wrote': SemanticQuaternion(x=0.0, y=-1.0, z=0.4, w=0.0),
    
    # Detective domain
    'holmes': SemanticQuaternion(x=1.0, y=1.0, z=1.0, w=1.0),    # male, adult, protagonist
    'watson': SemanticQuaternion(x=1.0, y=1.0, z=0.3, w=1.0),    # male, adult, helper
    'detective': SemanticQuaternion(x=0.0, y=1.0, z=0.8, w=1.0), # role, initiator
    'assistant': SemanticQuaternion(x=0.0, y=1.0, z=0.3, w=1.0), # role, helper
    'moriarty': SemanticQuaternion(x=1.0, y=1.0, z=0.8, w=1.0),  # male, adult, antagonist
    'villain': SemanticQuaternion(x=0.0, y=1.0, z=0.8, w=1.0),   # role, antagonist
    'lestrade': SemanticQuaternion(x=1.0, y=1.0, z=0.5, w=1.0),
    'inspector': SemanticQuaternion(x=0.0, y=1.0, z=0.5, w=1.0),
}


# =============================================================================
# SEMANTIC QUATERNION NAVIGATOR
# =============================================================================

class SemanticFeatureLearner:
    """
    Learn semantic features (x,y axes) from parallel structures.
    
    Just as morphology learns verb equivalence from parallel sentences:
        "I walk" / "I walked" → walk ≡ walked (same verb, different tense)
    
    We learn semantic features from parallel structures:
        "The king rules" / "The queen rules" → king vs queen differ only in gender
        "The man works" / "The boy plays" → man vs boy differ in age
    
    Key insight: If two words appear in the SAME SYNTACTIC POSITION
    with the SAME ACTION, they likely differ only in semantic features.
    """
    
    def __init__(self):
        # Learned feature differences: (word1, word2) -> (dx, dy)
        self.learned_pairs: Dict[Tuple[str, str], Tuple[float, float]] = {}
        
        # Context patterns: word -> {(action, position): count}
        self.context_patterns: Dict[str, Dict[Tuple[str, float], int]] = defaultdict(
            lambda: defaultdict(int)
        )
    
    def observe_sentence(self, sentence: str, initiator: str, mediator: str):
        """
        Observe a sentence to build context patterns.
        
        Records that 'initiator' appeared with 'mediator' action.
        """
        initiator_lower = initiator.lower()
        mediator_lower = mediator.lower()
        
        # Record this context
        self.context_patterns[initiator_lower][(mediator_lower, 0.0)] += 1
    
    def learn_from_parallel(self, word1: str, word2: str, 
                            shared_action: str) -> Optional[Tuple[float, float]]:
        """
        Learn feature difference from parallel structures.
        
        If word1 and word2 both appear with the same action,
        they likely differ only in semantic features.
        
        Returns (dx, dy) - the learned feature difference.
        """
        w1 = word1.lower()
        w2 = word2.lower()
        action = shared_action.lower()
        
        # Check if both words appear with this action
        w1_has_action = (action, 0.0) in self.context_patterns[w1]
        w2_has_action = (action, 0.0) in self.context_patterns[w2]
        
        if not (w1_has_action and w2_has_action):
            return None
        
        # They share an action - learn the difference
        # For now, use heuristics based on word properties
        dx, dy = self._infer_feature_difference(w1, w2)
        
        self.learned_pairs[(w1, w2)] = (dx, dy)
        self.learned_pairs[(w2, w1)] = (-dx, -dy)  # Symmetric
        
        return (dx, dy)
    
    def _infer_feature_difference(self, word1: str, word2: str) -> Tuple[float, float]:
        """
        Infer feature difference from word properties.
        
        Uses suffix patterns and known word lists.
        """
        # Gender suffixes
        gender_pairs = [
            ('ess', ''),      # actress/actor
            ('ress', 'r'),    # waitress/waiter  
            ('ine', ''),      # heroine/hero
        ]
        
        for fem_suffix, masc_suffix in gender_pairs:
            if word1.endswith(fem_suffix) and word2.endswith(masc_suffix):
                return (-2.0, 0.0)  # word1 is female
            if word2.endswith(fem_suffix) and word1.endswith(masc_suffix):
                return (2.0, 0.0)   # word1 is male
        
        # Known gender pairs
        gender_map = {
            ('king', 'queen'): (2.0, 0.0),
            ('man', 'woman'): (2.0, 0.0),
            ('boy', 'girl'): (2.0, 0.0),
            ('father', 'mother'): (2.0, 0.0),
            ('son', 'daughter'): (2.0, 0.0),
            ('prince', 'princess'): (2.0, 0.0),
            ('actor', 'actress'): (2.0, 0.0),
            ('waiter', 'waitress'): (2.0, 0.0),
            ('host', 'hostess'): (2.0, 0.0),
        }
        
        if (word1, word2) in gender_map:
            return gender_map[(word1, word2)]
        if (word2, word1) in gender_map:
            dx, dy = gender_map[(word2, word1)]
            return (-dx, -dy)
        
        # Age pairs
        age_map = {
            ('man', 'boy'): (0.0, 2.0),
            ('woman', 'girl'): (0.0, 2.0),
            ('dog', 'puppy'): (0.0, 2.0),
            ('cat', 'kitten'): (0.0, 2.0),
        }
        
        if (word1, word2) in age_map:
            return age_map[(word1, word2)]
        if (word2, word1) in age_map:
            dx, dy = age_map[(word2, word1)]
            return (-dx, -dy)
        
        # Default: no difference detected
        return (0.0, 0.0)
    
    def get_learned_difference(self, word1: str, word2: str) -> Optional[Tuple[float, float]]:
        """Get previously learned feature difference."""
        return self.learned_pairs.get((word1.lower(), word2.lower()))


class SemanticQuaternionNavigator:
    """
    Navigate concept space using semantic quaternions.
    
    Key insight: Analogies are ROTATIONS in 4D semantic space.
    
    king → queen = flip x-axis (gender)
    man → boy = shift y-axis (age)
    france → paris = shift z-axis (country → capital)
    
    Integration with Geometric System:
    - z-axis (agency) comes from learned φ-direction
    - w-axis (animacy) inferred from role counts
    - x,y axes can be learned from parallel structures
    
    Learning x,y from parallel structures:
    - "The king rules" + "The queen rules" → king/queen differ in x (gender)
    - "The man works" + "The boy plays" → man/boy differ in y (age)
    """
    
    def __init__(self, knowledge=None, use_defaults: bool = True):
        """
        Initialize with optional GeometricKnowledge.
        
        Args:
            knowledge: GeometricKnowledge instance for learned structure
            use_defaults: Whether to load default semantic features
        
        If knowledge is provided, z-axis (agency) is populated from φ-direction.
        Other axes use defaults or can be learned from parallel structures.
        """
        self.knowledge = knowledge
        self.concepts: Dict[str, SemanticQuaternion] = {}
        self.relations: Dict[str, SemanticQuaternion] = {}
        self.feature_learner = SemanticFeatureLearner()
        
        # Load defaults if requested
        if use_defaults:
            self.concepts.update(DEFAULT_SEMANTIC_FEATURES)
        
        # If we have knowledge, integrate geometric properties
        if knowledge:
            self._integrate_geometric_knowledge()
            self._learn_features_from_knowledge()
    
    def _integrate_geometric_knowledge(self):
        """
        Integrate learned geometric properties into quaternions.
        
        For each concept in knowledge:
        - z-axis = φ-direction (learned agency)
        - w-axis = animacy heuristic from role counts
        """
        if not self.knowledge:
            return
        
        for word, concept in self.knowledge.concepts.items():
            word_lower = word.lower()
            
            # Get existing quaternion or create new
            if word_lower in self.concepts:
                q = self.concepts[word_lower]
                # Update z from φ-direction (override default)
                q.z = concept.phi_direction
            else:
                # Create new quaternion from geometric properties
                total_roles = concept.initiator_count + concept.mediator_count + concept.receiver_count
                
                # Animacy heuristic: entities with roles are more animate
                animacy = 0.0
                if total_roles > 0:
                    if concept.initiator_count > 0:
                        animacy = 0.8  # Initiators are likely animate
                    elif concept.receiver_count > 0:
                        animacy = 0.3  # Receivers might be objects
                    if concept.mediator_count > 0:
                        animacy = 0.0  # Verbs are not animate
                
                self.concepts[word_lower] = SemanticQuaternion(
                    x=0.0,  # Unknown gender (could learn from context)
                    y=0.0,  # Unknown age (could learn from context)
                    z=concept.phi_direction,  # Agency from φ-direction!
                    w=animacy
                )
    
    def _learn_features_from_knowledge(self):
        """
        Learn x,y features from parallel structures in knowledge.
        
        Looks for pairs of words that share actions (parallel structures)
        and learns their feature differences.
        """
        if not self.knowledge:
            return
        
        # Build action → initiators map
        action_to_initiators: Dict[str, Set[str]] = defaultdict(set)
        
        for frame in self.knowledge.frames:
            if frame.initiator and frame.mediator:
                action_to_initiators[frame.mediator.lower()].add(frame.initiator.lower())
                # Record in feature learner
                self.feature_learner.observe_sentence(
                    frame.text or "",
                    frame.initiator,
                    frame.mediator
                )
        
        # Find pairs that share actions (parallel structures)
        for action, initiators in action_to_initiators.items():
            if len(initiators) < 2:
                continue
            
            initiator_list = list(initiators)
            for i, w1 in enumerate(initiator_list):
                for w2 in initiator_list[i+1:]:
                    # These two words share an action - learn difference
                    diff = self.feature_learner.learn_from_parallel(w1, w2, action)
                    
                    if diff and (diff[0] != 0 or diff[1] != 0):
                        # Update quaternions with learned features
                        self._apply_learned_difference(w1, w2, diff)
    
    def _apply_learned_difference(self, word1: str, word2: str, diff: Tuple[float, float]):
        """
        Apply learned feature difference to quaternions.
        
        If we learn that word1 and word2 differ by (dx, dy),
        update their quaternions accordingly.
        """
        dx, dy = diff
        
        # Get or create quaternions
        if word1 not in self.concepts:
            self.add_concept(word1)
        if word2 not in self.concepts:
            self.add_concept(word2)
        
        q1 = self.concepts[word1]
        q2 = self.concepts[word2]
        
        # If both have default x,y (0,0), set them based on difference
        if q1.x == 0 and q1.y == 0 and q2.x == 0 and q2.y == 0:
            # Center them around 0
            q1.x = dx / 2
            q1.y = dy / 2
            q2.x = -dx / 2
            q2.y = -dy / 2
    
    def add_concept(self, word: str, q: SemanticQuaternion = None):
        """
        Add a concept with optional quaternion.
        
        If no quaternion provided, tries to infer from knowledge or defaults.
        """
        word_lower = word.lower()
        
        if q is not None:
            self.concepts[word_lower] = q
        elif word_lower in self.concepts:
            pass  # Already have it
        elif word_lower in DEFAULT_SEMANTIC_FEATURES:
            self.concepts[word_lower] = DEFAULT_SEMANTIC_FEATURES[word_lower]
        elif self.knowledge and word_lower in self.knowledge.concepts:
            # Infer from geometric properties
            c = self.knowledge.concepts[word_lower]
            total_roles = c.initiator_count + c.mediator_count + c.receiver_count
            animacy = 0.8 if c.initiator_count > 0 else 0.3 if total_roles > 0 else 0.0
            
            self.concepts[word_lower] = SemanticQuaternion(
                x=0.0,  # Unknown gender
                y=0.0,  # Unknown age
                z=c.phi_direction,  # Agency from φ-direction!
                w=animacy
            )
        else:
            # Default neutral
            self.concepts[word_lower] = SemanticQuaternion(0, 0, 0, 0)
    
    def add_concepts(self, words: List[str]):
        """Add multiple concepts."""
        for word in words:
            self.add_concept(word)
    
    def get_quaternion(self, word: str) -> SemanticQuaternion:
        """Get quaternion for a word."""
        word_lower = word.lower()
        if word_lower in self.concepts:
            return self.concepts[word_lower]
        
        # Try to add it
        self.add_concept(word)
        return self.concepts.get(word_lower, SemanticQuaternion(0, 0, 0, 0))
    
    def similarity(self, word1: str, word2: str) -> float:
        """Compute semantic similarity via quaternion cosine."""
        q1 = self.get_quaternion(word1)
        q2 = self.get_quaternion(word2)
        return q1.cosine_similarity(q2)
    
    def extract_relation(self, a: str, b: str) -> SemanticQuaternion:
        """
        Extract the relation (rotation) from A to B.
        
        This is the quaternion difference: B - A
        """
        q_a = self.get_quaternion(a)
        q_b = self.get_quaternion(b)
        return q_b - q_a
    
    def add_relation(self, name: str, a: str, b: str):
        """Learn a named relation from an example pair."""
        self.relations[name.lower()] = self.extract_relation(a, b)
    
    def complete_analogy(self, a: str, b: str, c: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Complete analogy: A is to B as C is to ?
        
        Uses quaternion arithmetic: ? = C + (B - A)
        
        The relation (B - A) is a "rotation" in semantic space.
        """
        q_a = self.get_quaternion(a)
        q_b = self.get_quaternion(b)
        q_c = self.get_quaternion(c)
        
        # Relation quaternion (the "rotation")
        relation = q_b - q_a
        
        # Apply to C
        q_target = q_c + relation
        
        # Find closest concepts
        exclude = {a.lower(), b.lower(), c.lower()}
        return self.find_k_closest(q_target, k, exclude)
    
    def apply_relation(self, relation_name: str, word: str, k: int = 5) -> List[Tuple[str, float]]:
        """Apply a learned relation to a word."""
        if relation_name.lower() not in self.relations:
            return []
        
        relation = self.relations[relation_name.lower()]
        q_word = self.get_quaternion(word)
        q_target = q_word + relation
        
        exclude = {word.lower()}
        return self.find_k_closest(q_target, k, exclude)
    
    def find_k_closest(self, q: SemanticQuaternion, k: int = 5,
                       exclude: Set[str] = None) -> List[Tuple[str, float]]:
        """Find k closest concepts to a target quaternion."""
        if exclude is None:
            exclude = set()
        
        distances = []
        for word, q_word in self.concepts.items():
            if word in exclude:
                continue
            
            distance = q.distance(q_word)
            distances.append((word, distance))
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def find_similar_relations(self, a: str, b: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """Find pairs with similar relations to A:B."""
        target_relation = self.extract_relation(a, b)
        target_mag = target_relation.magnitude
        
        if target_mag == 0:
            return []
        
        similar = []
        words = list(self.concepts.keys())
        
        for i, w1 in enumerate(words):
            for w2 in words[i+1:]:
                if (w1 == a.lower() and w2 == b.lower()) or \
                   (w1 == b.lower() and w2 == a.lower()):
                    continue
                
                relation = self.get_quaternion(w2) - self.get_quaternion(w1)
                sim = target_relation.cosine_similarity(relation)
                similar.append((w1, w2, sim))
        
        similar.sort(key=lambda x: -x[2])
        return similar[:k]
    
    def visualize(self, word: str) -> str:
        """Visualize a word's semantic quaternion."""
        q = self.get_quaternion(word)
        
        gender = "male" if q.x > 0.3 else "female" if q.x < -0.3 else "neutral"
        age = "adult" if q.y > 0.3 else "young" if q.y < -0.3 else "neutral"
        agency = "initiator" if q.z > 0.3 else "receiver" if q.z < -0.3 else "neutral"
        animacy = "human" if q.w > 0.5 else "animal" if q.w > 0 else "place/abstract"
        
        return f"{word}: {q} → {gender}, {age}, {agency}, {animacy}"


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate semantic quaternion navigation."""
    print("=" * 70)
    print("SEMANTIC QUATERNION NAVIGATOR DEMO")
    print("=" * 70)
    
    navigator = SemanticQuaternionNavigator()
    
    # Show some encodings
    print("\n" + "-" * 70)
    print("SEMANTIC QUATERNION ENCODINGS")
    print("-" * 70)
    
    sample_words = ["king", "queen", "man", "woman", "boy", "girl",
                    "france", "paris", "dog", "puppy", "walk", "walked"]
    
    for word in sample_words:
        print(f"  {navigator.visualize(word)}")
    
    # Test analogies
    print("\n" + "-" * 70)
    print("ANALOGY COMPLETION (Quaternion Rotation)")
    print("-" * 70)
    
    analogies = [
        ("king", "queen", "man"),       # man → woman (gender flip)
        ("man", "woman", "boy"),        # boy → girl (gender flip)
        ("father", "mother", "son"),    # son → daughter (gender flip)
        ("actor", "actress", "waiter"), # waiter → waitress (gender flip)
        ("france", "paris", "germany"), # germany → berlin (country → capital)
        ("japan", "tokyo", "italy"),    # italy → rome (country → capital)
        ("walk", "walked", "run"),      # run → ran (tense flip)
        ("speak", "spoke", "write"),    # write → wrote (tense flip)
        ("dog", "puppy", "cat"),        # cat → kitten (age flip)
        ("holmes", "detective", "watson"),  # watson → assistant (role)
    ]
    
    correct = 0
    total = len(analogies)
    
    expected = {
        ("king", "queen", "man"): "woman",
        ("man", "woman", "boy"): "girl",
        ("father", "mother", "son"): "daughter",
        ("actor", "actress", "waiter"): "waitress",
        ("france", "paris", "germany"): "berlin",
        ("japan", "tokyo", "italy"): "rome",
        ("walk", "walked", "run"): "ran",
        ("speak", "spoke", "write"): "wrote",
        ("dog", "puppy", "cat"): "kitten",
        ("holmes", "detective", "watson"): "assistant",
    }
    
    for a, b, c in analogies:
        results = navigator.complete_analogy(a, b, c, k=3)
        top_answers = [r[0] for r in results]
        
        print(f"\n  {a} : {b} :: {c} : ?")
        
        # Show the relation
        relation = navigator.extract_relation(a, b)
        print(f"    Relation: Δx={relation.x:.1f}, Δy={relation.y:.1f}, Δz={relation.z:.1f}, Δw={relation.w:.1f}")
        print(f"    Top answers: {top_answers}")
        
        exp = expected.get((a, b, c), "?")
        if exp in top_answers:
            print(f"    ✓ Expected '{exp}' found!")
            correct += 1
        else:
            print(f"    ✗ Expected '{exp}'")
    
    print(f"\n  Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")
    
    # Test similarity
    print("\n" + "-" * 70)
    print("SEMANTIC SIMILARITY (Quaternion Cosine)")
    print("-" * 70)
    
    pairs = [
        ("king", "queen"),    # Same except gender
        ("king", "prince"),   # Same gender, different age
        ("king", "man"),      # Same gender, different royalty
        ("king", "dog"),      # Very different
        ("france", "germany"), # Both countries
        ("france", "paris"),  # Country vs capital
    ]
    
    for w1, w2 in pairs:
        sim = navigator.similarity(w1, w2)
        print(f"  {w1} vs {w2}: {sim:.3f}")
    
    # Find similar relations
    print("\n" + "-" * 70)
    print("SIMILAR RELATIONS (Same Rotation)")
    print("-" * 70)
    
    print("\nPairs with similar relation to 'king' → 'queen' (gender flip):")
    similar = navigator.find_similar_relations("king", "queen", k=5)
    for w1, w2, sim in similar:
        print(f"  {w1} → {w2}: {sim:.3f}")
    
    print("\nPairs with similar relation to 'dog' → 'puppy' (age flip):")
    similar = navigator.find_similar_relations("dog", "puppy", k=5)
    for w1, w2, sim in similar:
        print(f"  {w1} → {w2}: {sim:.3f}")


if __name__ == "__main__":
    demo()
