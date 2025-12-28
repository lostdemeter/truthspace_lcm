#!/usr/bin/env python3
"""
Diffraction Grating LCM

Key insight: Instead of complex holographic calculations, use the SAME
architecture from two orthogonal viewpoints. The interference pattern
emerges from the geometry of overlap.

A diffraction grating works by:
1. Light passes through multiple slits
2. Each slit creates a wavefront
3. Wavefronts interfere based on path difference
4. Constructive interference where paths align

For concepts:
1. View 1 (horizontal): Actor → Action → Target (narrative flow)
2. View 2 (vertical): Domain → Role → Relationship (structural view)
3. Interference: Concepts that align in BOTH views reinforce
4. Result: Natural filtering without explicit phase calculation

The "slits" are our frames, the "path difference" is the φ-direction,
and the "interference pattern" is the query result.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import re
import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.morphological_quaternion import MorphologicalTransformer

PHI = 1.618034


@dataclass
class GratingConcept:
    """
    A concept with two orthogonal encodings (the "grating").
    
    View 1 (Horizontal/Narrative): How the concept flows in sentences
      - Position in frame (actor=0, action=0.5, target=1)
      - φ-direction (entity vs action)
      
    View 2 (Vertical/Structural): How the concept relates structurally
      - Domain membership
      - Role type (protagonist, antagonist, object, etc.)
    """
    word: str
    
    # View 1: Narrative position (horizontal slit)
    narrative_position: float = 0.5  # 0=actor, 0.5=action, 1=target
    phi_direction: float = 0.0       # +1=entity, -1=action
    
    # View 2: Structural position (vertical slit)
    domain_position: float = 0.0     # Which domain (0-1 normalized)
    role_position: float = 0.5       # Role type (0=protagonist, 0.5=supporting, 1=object)
    
    # Counts for computing positions
    actor_count: int = 0
    action_count: int = 0
    target_count: int = 0
    domains: Counter = field(default_factory=Counter)
    
    # Relationships
    actions_performed: Counter = field(default_factory=Counter)
    targets_hit: Counter = field(default_factory=Counter)
    
    def compute_positions(self, domain_map: Dict[str, float]):
        """Compute both view positions from counts."""
        # View 1: Narrative position
        total = self.actor_count + self.action_count + self.target_count
        if total > 0:
            # Weighted average: actor=0, action=0.5, target=1
            self.narrative_position = (
                self.actor_count * 0.0 +
                self.action_count * 0.5 +
                self.target_count * 1.0
            ) / total
            
            # φ-direction: entity (+) vs action (-)
            entity_count = self.actor_count + self.target_count
            self.phi_direction = (entity_count - self.action_count) / total
        
        # View 2: Structural position
        if self.domains:
            # Domain position from primary domain
            primary = self.domains.most_common(1)[0][0]
            self.domain_position = domain_map.get(primary, 0.5)
            
            # Role position: protagonist (many actions) vs object (target only)
            if self.actor_count > self.target_count:
                self.role_position = 0.0  # Protagonist
            elif self.target_count > self.actor_count:
                self.role_position = 1.0  # Object
            else:
                self.role_position = 0.5  # Supporting
    
    def interference_with(self, other: 'GratingConcept') -> float:
        """
        Compute interference between two concepts using both views.
        
        Like a diffraction grating:
        - View 1 alignment contributes to interference
        - View 2 alignment contributes to interference
        - Both must align for strong constructive interference
        
        Returns: -1 (destructive) to +1 (constructive)
        """
        # View 1: Narrative alignment
        # Complementary positions interfere constructively (actor+target)
        narrative_diff = abs(self.narrative_position - other.narrative_position)
        
        # For actor-target pairs (diff ~1.0), we want constructive
        # For same-role pairs (diff ~0), we want neutral/destructive
        narrative_alignment = narrative_diff - 0.5  # -0.5 to +0.5
        
        # φ-direction: opposite directions attract (polyomino fitting)
        # This is the key insight from our symmetric understanding
        phi_product = self.phi_direction * other.phi_direction
        phi_alignment = -phi_product  # Opposite = constructive (+1), same = destructive (-1)
        
        # View 2: Structural alignment
        # Same domain = constructive (they belong together)
        domain_diff = abs(self.domain_position - other.domain_position)
        domain_alignment = 1.0 - 2 * domain_diff  # 1 at same, -1 at opposite
        
        # Complementary roles = constructive (protagonist acts on object)
        role_diff = abs(self.role_position - other.role_position)
        role_alignment = role_diff  # 0 at same role, 1 at complementary
        
        # Combined interference
        # View 1: narrative flow (how they connect in sentences)
        view1 = (narrative_alignment + phi_alignment) / 2
        
        # View 2: structural fit (where they belong)
        view2 = (domain_alignment + role_alignment) / 2
        
        # Weighted combination - domain alignment is most important for filtering
        # but φ-alignment is most important for fitting
        combined = 0.4 * phi_alignment + 0.3 * domain_alignment + 0.2 * role_alignment + 0.1 * narrative_alignment
        
        return combined


@dataclass 
class GratingFrame:
    """A frame with both view encodings."""
    actor: str
    action: str
    target: Optional[str]
    domain: str


# Domain definitions
DOMAINS = {
    'sherlock': {'position': 0.0, 'genre': 'mystery'},
    'alice': {'position': 0.2, 'genre': 'fantasy'},
    'pride': {'position': 0.4, 'genre': 'romance'},
    'gatsby': {'position': 0.6, 'genre': 'tragedy'},
    'hamlet': {'position': 0.8, 'genre': 'tragedy'},
    'unknown': {'position': 0.5, 'genre': 'general'},
}

DOMAIN_KEYWORDS = {
    'sherlock': {'holmes', 'watson', 'detective', 'lestrade', 'moriarty', 'mycroft', 'hudson'},
    'alice': {'alice', 'rabbit', 'queen', 'cheshire', 'hatter', 'dormouse', 'caterpillar', 'wonderland'},
    'pride': {'darcy', 'elizabeth', 'bennet', 'bingley', 'wickham', 'jane', 'lydia', 'longbourn'},
    'gatsby': {'gatsby', 'nick', 'daisy', 'tom', 'wilson', 'myrtle', 'buchanan', 'carraway'},
    'hamlet': {'hamlet', 'claudius', 'gertrude', 'ophelia', 'polonius', 'laertes', 'horatio', 'ghost'},
}


class DiffractionGratingLCM:
    """
    Language model using diffraction grating interference.
    
    The key simplification: instead of computing complex phases,
    we use the SAME architecture from two orthogonal viewpoints
    and let the geometry create the interference pattern.
    
    View 1 (Horizontal): Narrative flow (actor → action → target)
    View 2 (Vertical): Structural role (domain, protagonist/object)
    
    Query = "slit pattern"
    Result = "interference pattern" where both views align
    """
    
    def __init__(self):
        self.concepts: Dict[str, GratingConcept] = {}
        self.frames: List[GratingFrame] = []
        self.morpho = MorphologicalTransformer()
        self.domain_map = {d: info['position'] for d, info in DOMAINS.items()}
        
        self.function_words = {
            'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
            'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from',
            'he', 'she', 'it', 'they', 'his', 'her', 'its', 'their',
            'that', 'this', 'these', 'those', 'which', 'who', 'whom',
            'and', 'or', 'but', 'if', 'then', 'so', 'as', 'than',
            'very', 'more', 'most', 'down', 'up', 'out', 'about',
            'had', 'has', 'have', 'did', 'do', 'does', 'would', 'could', 'should',
            'not', 'no', 'yes', 'all', 'some', 'any', 'each', 'every',
        }
    
    def _detect_domain(self, text: str) -> str:
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        best_domain = 'unknown'
        best_score = 0
        
        for domain, keywords in DOMAIN_KEYWORDS.items():
            overlap = len(words & keywords)
            if overlap > best_score:
                best_score = overlap
                best_domain = domain
        
        return best_domain
    
    def _get_or_create(self, word: str) -> GratingConcept:
        word_lower = word.lower()
        if word_lower not in self.concepts:
            self.concepts[word_lower] = GratingConcept(word=word_lower)
        return self.concepts[word_lower]
    
    def _extract_frame(self, sentence: str, domain: str) -> Optional[GratingFrame]:
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        content = [t for t in tokens if t not in self.function_words and len(t) > 2]
        
        if len(content) < 2:
            return None
        
        actor = content[0]
        action = content[1]
        target = content[2] if len(content) > 2 else None
        
        if target and (target.endswith('ly') or len(target) <= 3):
            target = content[3] if len(content) > 3 else None
        
        return GratingFrame(actor=actor, action=action, target=target, domain=domain)
    
    def ingest(self, text: str):
        """Ingest text and build the diffraction grating."""
        paragraphs = re.split(r'\n\s*\n', text)
        
        for para in paragraphs:
            domain = self._detect_domain(para)
            sentences = re.split(r'[.!?]+', para)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sent_domain = self._detect_domain(sentence)
                if sent_domain != 'unknown':
                    domain = sent_domain
                
                frame = self._extract_frame(sentence, domain)
                if frame:
                    self.frames.append(frame)
                    self._update_concepts(frame)
        
        # Compute positions for all concepts
        for concept in self.concepts.values():
            concept.compute_positions(self.domain_map)
    
    def _update_concepts(self, frame: GratingFrame):
        actor = self._get_or_create(frame.actor)
        actor.actor_count += 1
        actor.domains[frame.domain] += 1
        actor.actions_performed[frame.action] += 1
        
        action = self._get_or_create(frame.action)
        action.action_count += 1
        action.domains[frame.domain] += 1
        
        if frame.target:
            target = self._get_or_create(frame.target)
            target.target_count += 1
            target.domains[frame.domain] += 1
            actor.targets_hit[frame.target] += 1
    
    def query(self, query_text: str, n: int = 5) -> List[Tuple[str, float]]:
        """
        Query using diffraction grating interference.
        
        The query creates a "slit pattern" and we find concepts
        that create constructive interference with it.
        """
        query_lower = query_text.lower()
        query_domain = self._detect_domain(query_text)
        
        # Extract query words (excluding function words and question words)
        query_words = set(re.findall(r'\b\w+\b', query_lower)) - self.function_words
        query_words -= {'who', 'what', 'does', 'is', 'do', 'did', 'how', 'why', 'where', 'when', 'tell', 'me', 'about'}
        
        # Detect what type of answer we're looking for
        looking_for_actor = 'who' in query_lower
        looking_for_action = 'what' in query_lower and 'do' in query_lower
        looking_for_target = 'what' in query_lower and 'do' not in query_lower
        
        # Find action being queried (e.g., "who watches" -> watch)
        query_action = None
        for qw in query_words:
            base = self.morpho._get_base(qw)
            # Check if this is an action in our corpus
            if base in self.concepts and self.concepts[base].action_count > 0:
                query_action = base
                break
            # Also check the word itself
            if qw in self.concepts and self.concepts[qw].action_count > 0:
                query_action = qw
                break
        
        results = []
        for name, concept in self.concepts.items():
            score = 0.0
            
            # Domain filtering - boost same domain
            if query_domain != 'unknown':
                if query_domain in [d for d, _ in concept.domains.most_common(3)]:
                    score += 0.3
            
            # Type filtering based on question type
            if looking_for_actor:
                if concept.actor_count > 0:
                    score += 0.5
                    # If asking "who [action]s", check if this actor does that action
                    if query_action:
                        for act in concept.actions_performed:
                            if self.morpho._get_base(act) == query_action:
                                score += 1.0
                                break
            
            elif looking_for_action:
                if concept.action_count > 0:
                    score += 0.5
            
            elif looking_for_target:
                if concept.target_count > 0:
                    score += 0.5
            
            # Direct match boost
            if name in query_words:
                score += 0.8
            
            # Skip low scores
            if score > 0:
                results.append((name, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]
    
    def find_targets_for(self, actor: str, action: str = None) -> List[Tuple[str, float]]:
        """
        Find targets using diffraction interference.
        
        The actor+action create a "slit pattern" and we find targets
        that constructively interfere.
        """
        if actor not in self.concepts:
            return []
        
        actor_concept = self.concepts[actor]
        
        # If action specified, use it; otherwise use most common
        if not action and actor_concept.actions_performed:
            action = actor_concept.actions_performed.most_common(1)[0][0]
        
        if not action:
            return []
        
        action_concept = self.concepts.get(action)
        if not action_concept:
            return []
        
        # Find targets that interfere constructively with actor+action
        results = []
        for name, concept in self.concepts.items():
            if name == actor or name == action:
                continue
            
            # Must be target-like (narrative position > 0.5)
            if concept.narrative_position < 0.3:
                continue
            
            # Interference with actor (should be complementary)
            actor_interference = actor_concept.interference_with(concept)
            
            # Interference with action (should be complementary)
            action_interference = action_concept.interference_with(concept)
            
            # Combined: both must be positive for strong result
            if actor_interference > 0 and action_interference > 0:
                combined = math.sqrt(actor_interference * action_interference)
            else:
                combined = 0.0
            
            # Boost if this target was actually used with this actor
            if name in actor_concept.targets_hit:
                combined += 0.5
            
            if combined > 0:
                results.append((name, combined))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def generate(self, seed: str, domain: str = None) -> str:
        """Generate text using diffraction interference."""
        seed_lower = seed.lower()
        
        if seed_lower not in self.concepts:
            return f"I don't have information about {seed}."
        
        concept = self.concepts[seed_lower]
        
        # Find action using interference
        if concept.actions_performed:
            action = concept.actions_performed.most_common(1)[0][0]
        else:
            return f"{seed.title()} appears in the text."
        
        # Find target using diffraction
        targets = self.find_targets_for(seed_lower, action)
        target = None
        for t, score in targets:
            if not t.endswith('ly') and len(t) > 3:
                target = t
                break
        
        # Conjugate
        from experiments.morphological_quaternion import MorphoQuaternion
        q3 = MorphoQuaternion(x=1, y=-1, z=0, w=-1)  # 3rd singular present simple
        verb = self.morpho.transform(self.morpho._get_base(action), q3)
        
        if target:
            return f"{seed.title()} {verb} {target}."
        return f"{seed.title()} {verb}."
    
    def describe(self, entity: str) -> str:
        """Describe an entity using both views."""
        entity_lower = entity.lower()
        
        if entity_lower not in self.concepts:
            return f"I don't have information about {entity}."
        
        concept = self.concepts[entity_lower]
        
        # View 1: Narrative description based on role
        # Prioritize actor role for characters
        if concept.actor_count > 0:
            if concept.actor_count >= concept.target_count:
                narrative_role = "protagonist"
            else:
                narrative_role = "character"
        elif concept.target_count > 0:
            narrative_role = "object"
        elif concept.action_count > 0:
            # Skip describing actions as entities
            return f"{entity.title()} is an action in the narrative."
        else:
            narrative_role = "concept"
        
        # View 2: Structural description
        primary_domain = concept.domains.most_common(1)[0][0] if concept.domains else 'unknown'
        genre = DOMAINS.get(primary_domain, {}).get('genre', 'general')
        
        # Actions performed
        actions = list(concept.actions_performed.keys())[:3]
        if actions:
            from experiments.morphological_quaternion import MorphoQuaternion
            q3 = MorphoQuaternion(x=1, y=-1, z=0, w=-1)  # 3rd singular present simple
            verbs = [self.morpho.transform(self.morpho._get_base(a), q3) for a in actions]
            action_str = f" who {', '.join(verbs)}"
        else:
            action_str = ""
        
        # Targets
        targets = list(concept.targets_hit.keys())[:2]
        target_str = ""
        if targets:
            good_targets = [t for t in targets if not t.endswith('ly') and len(t) > 3]
            if good_targets:
                target_str = f" (targets: {', '.join(good_targets)})"
        
        return f"{entity.title()} is a {narrative_role}{action_str}{target_str} in the {genre} genre."
    
    def cross_domain_query(self, action: str) -> Dict[str, List[str]]:
        """
        Find who performs an action across all domains.
        
        This is where the diffraction grating shines - we can see
        the same "pattern" (action) across different "slits" (domains).
        """
        base_action = self.morpho._get_base(action)
        
        by_domain = defaultdict(list)
        
        for name, concept in self.concepts.items():
            if concept.actor_count == 0:
                continue
            
            for act in concept.actions_performed:
                if self.morpho._get_base(act) == base_action:
                    primary = concept.domains.most_common(1)[0][0] if concept.domains else 'unknown'
                    by_domain[primary].append(name)
                    break
        
        return dict(by_domain)
    
    def get_interference_matrix(self, entities: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Get interference matrix between entities.
        
        This shows how concepts relate through both views.
        """
        matrix = {}
        for e1 in entities:
            if e1 not in self.concepts:
                continue
            matrix[e1] = {}
            c1 = self.concepts[e1]
            for e2 in entities:
                if e2 not in self.concepts:
                    continue
                c2 = self.concepts[e2]
                matrix[e1][e2] = c1.interference_with(c2)
        return matrix


# Test corpus
TEST_CORPUS = """
# Sherlock Holmes
Holmes examined the evidence carefully. Watson watched from the doorway.
The detective studied the footprints methodically. He noticed something unusual.
Holmes said to Watson that the case was elementary. Watson replied thoughtfully.
Lestrade questioned the witnesses thoroughly. Holmes observed the room.
Watson wrote in his journal diligently. Holmes deduced the identity brilliantly.
Moriarty plotted against Holmes secretly. The professor was cunning.

# Alice in Wonderland
Alice fell down the rabbit hole unexpectedly. She wondered where she was going.
The Queen shouted angrily at everyone. Alice felt confused and scared.
The Cheshire Cat smiled mysteriously. He disappeared slowly into thin air.
Alice grew very tall suddenly. She shrank very small moments later.
The Mad Hatter laughed wildly at the party. He poured more tea endlessly.

# Pride and Prejudice
Darcy looked at Elizabeth proudly. She ignored him completely at first.
Elizabeth danced gracefully at the ball. Darcy watched her intently.
Mr Bennet read his newspaper quietly. Mrs Bennet worried constantly.
Jane smiled sweetly at everyone. Bingley fell in love immediately.

# Hamlet
Hamlet pondered existence deeply. He questioned whether to live or die.
The ghost appeared to Hamlet mysteriously. He revealed the murder.
Claudius poisoned King Hamlet treacherously. He married Gertrude immediately.
Ophelia loved Hamlet devotedly. She went mad from grief eventually.
Hamlet killed Claudius finally. He avenged his father at last.
"""


def run_demo():
    """Demonstrate diffraction grating LCM."""
    print("=" * 70)
    print("DIFFRACTION GRATING LCM")
    print("=" * 70)
    print()
    print("Key insight: Same architecture, two orthogonal views.")
    print("Interference emerges from geometry, not complex calculations.")
    print()
    print("View 1 (Horizontal): Narrative flow (actor → action → target)")
    print("View 2 (Vertical):   Structural role (domain, protagonist/object)")
    print()
    
    # Create and ingest
    model = DiffractionGratingLCM()
    model.ingest(TEST_CORPUS)
    
    print(f"Learned {len(model.concepts)} concepts from {len(model.frames)} frames")
    print()
    
    # Show concept positions
    print("=" * 70)
    print("CONCEPT POSITIONS (The Grating)")
    print("=" * 70)
    print()
    print(f"{'Concept':<15} {'Narr.Pos':>10} {'φ-dir':>8} {'Domain':>10} {'Role':>8}")
    print("-" * 55)
    
    key_concepts = ['holmes', 'watson', 'alice', 'darcy', 'hamlet', 
                    'examined', 'watched', 'killed', 'loved', 'evidence']
    
    for name in key_concepts:
        if name in model.concepts:
            c = model.concepts[name]
            domain = c.domains.most_common(1)[0][0] if c.domains else 'unknown'
            print(f"{name:<15} {c.narrative_position:>10.2f} {c.phi_direction:>8.2f} {domain:>10} {c.role_position:>8.2f}")
    print()
    
    # Show interference between key entities
    print("=" * 70)
    print("INTERFERENCE MATRIX (Constructive = positive)")
    print("=" * 70)
    print()
    
    entities = ['holmes', 'watson', 'alice', 'darcy', 'hamlet']
    matrix = model.get_interference_matrix(entities)
    
    print(f"{'':>10}", end='')
    for e in entities:
        print(f"{e:>10}", end='')
    print()
    
    for e1 in entities:
        if e1 not in matrix:
            continue
        print(f"{e1:>10}", end='')
        for e2 in entities:
            if e2 in matrix[e1]:
                val = matrix[e1][e2]
                print(f"{val:>10.2f}", end='')
            else:
                print(f"{'--':>10}", end='')
        print()
    print()
    
    # Cross-domain queries
    print("=" * 70)
    print("CROSS-DOMAIN QUERIES (Same pattern, different slits)")
    print("=" * 70)
    print()
    
    for action in ['watch', 'kill', 'love', 'examine']:
        result = model.cross_domain_query(action)
        if result:
            print(f"Who {action}s?")
            for domain, actors in result.items():
                genre = DOMAINS.get(domain, {}).get('genre', domain)
                print(f"  {genre}: {', '.join([a.title() for a in actors])}")
            print()
    
    # Target finding using diffraction
    print("=" * 70)
    print("TARGET FINDING (Diffraction interference)")
    print("=" * 70)
    print()
    
    for actor in ['holmes', 'alice', 'hamlet']:
        targets = model.find_targets_for(actor)
        if targets:
            print(f"{actor.title()}'s targets:")
            for t, score in targets[:3]:
                print(f"  {t}: {score:.2f}")
            print()
    
    # Entity descriptions
    print("=" * 70)
    print("ENTITY DESCRIPTIONS (Both views)")
    print("=" * 70)
    print()
    
    for entity in ['holmes', 'watson', 'alice', 'darcy', 'hamlet', 'ophelia']:
        print(model.describe(entity))
    print()
    
    # Query examples
    print("=" * 70)
    print("QUERIES (Interference patterns)")
    print("=" * 70)
    print()
    
    queries = [
        "Who examines in Sherlock Holmes?",
        "Who watches?",
        "What does Holmes do?",
        "Who kills in Hamlet?",
    ]
    
    for q in queries:
        print(f"Q: {q}")
        results = model.query(q, n=3)
        if results:
            print(f"A: {', '.join([f'{r[0]} ({r[1]:.2f})' for r in results])}")
        else:
            print("A: No results")
        print()
    
    return model


if __name__ == "__main__":
    model = run_demo()
