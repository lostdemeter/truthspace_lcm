#!/usr/bin/env python3
"""
Geodesic Generator: Free-Form Text via Concept Space Navigation

Instead of template-based generation, this module generates text by
navigating through concept space along geodesic paths.

Key Insight: A coherent response is a PATH through concept space.
- Start at the query concept
- Navigate to related concepts via φ-weighted edges
- Generate a sentence at each waypoint
- The path structure IS the response structure

Mathematical Foundation:
- Concepts are points in φ-space (position from geometric knowledge)
- Edges are weighted by co-occurrence and φ-direction alignment
- Geodesic = shortest path that maintains semantic coherence
- Each step generates one sentence fragment

This replaces template slot-filling with continuous navigation.

Author: Lesley Gushurst
License: GPLv3
"""

import math
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, TYPE_CHECKING
from collections import defaultdict


PHI = 1.618034


@dataclass
class ConceptNode:
    """A concept in the navigation graph."""
    name: str
    phi_direction: float = 0.0  # Agency: initiator (+) vs receiver (-)
    mean_position: float = 0.5  # Typical sentence position
    importance: float = 1.0     # How central this concept is
    
    # Connections to other concepts
    neighbors: Dict[str, float] = field(default_factory=dict)  # name -> weight
    
    # What this concept does/receives
    actions: Dict[str, int] = field(default_factory=dict)  # verb -> count
    targets: Dict[str, int] = field(default_factory=dict)  # object -> count
    sources: Dict[str, int] = field(default_factory=dict)  # subject -> count


@dataclass
class GeodesicPath:
    """A path through concept space."""
    nodes: List[str]
    total_weight: float
    sentences: List[str] = field(default_factory=list)


class GeodesicGenerator:
    """
    Generate free-form text by navigating concept space.
    
    The key insight: coherent text follows geodesic paths through
    concept space. Instead of filling slots, we TRAVERSE.
    
    Navigation principles:
    1. Start at query concept
    2. Move to related concepts (weighted by co-occurrence)
    3. Prefer concepts with compatible φ-direction
    4. Generate sentence at each waypoint
    5. Stop when we've covered enough ground
    """
    
    def __init__(self, knowledge=None):
        """
        Initialize with optional GeometricKnowledge.
        
        Args:
            knowledge: GeometricKnowledge for concept graph
        """
        self.nodes: Dict[str, ConceptNode] = {}
        self.knowledge = knowledge
        
        if knowledge:
            self._build_graph_from_knowledge(knowledge)
    
    def _build_graph_from_knowledge(self, knowledge):
        """Build navigation graph from GeometricKnowledge."""
        # Create nodes for all content words
        for name, concept in knowledge.concepts.items():
            if concept.is_content_word:
                node = ConceptNode(
                    name=name,
                    phi_direction=concept.phi_direction,
                    mean_position=concept.mean_position,
                    importance=self._compute_importance(concept),
                )
                
                # Copy actions and targets
                node.actions = dict(concept.actions)
                node.targets = dict(concept.targets)
                
                self.nodes[name] = node
        
        # Build edges from frame co-occurrence
        for frame in knowledge.frames:
            # Frame is a dataclass, not a dict
            initiator = (frame.initiator or '').lower()
            mediator = (frame.mediator or '').lower()
            receiver = (frame.receiver or '').lower()
            
            # Connect initiator ↔ mediator ↔ receiver
            pairs = [
                (initiator, mediator),
                (mediator, receiver),
                (initiator, receiver),
            ]
            
            for a, b in pairs:
                if a and b and a in self.nodes and b in self.nodes:
                    # Weight by φ-direction compatibility
                    phi_a = self.nodes[a].phi_direction
                    phi_b = self.nodes[b].phi_direction
                    
                    # Complementary directions (initiator→receiver) get higher weight
                    compatibility = 1.0 + abs(phi_a - phi_b) / 2
                    
                    # Add bidirectional edge
                    self.nodes[a].neighbors[b] = self.nodes[a].neighbors.get(b, 0) + compatibility
                    self.nodes[b].neighbors[a] = self.nodes[b].neighbors.get(a, 0) + compatibility
                    
                    # Track sources for receiver
                    if a == initiator and b == receiver:
                        self.nodes[b].sources[a] = self.nodes[b].sources.get(a, 0) + 1
    
    def _compute_importance(self, concept) -> float:
        """Compute concept importance using φ-weighting."""
        total_roles = concept.initiator_count + concept.mediator_count + concept.receiver_count
        if total_roles == 0:
            return 0.1
        
        # More roles = more important, scaled by φ
        return min(PHI, 0.5 + total_roles * 0.1)
    
    def find_geodesic(self, start: str, end: str, max_length: int = 5) -> Optional[GeodesicPath]:
        """
        Find geodesic path between two concepts.
        
        Uses Dijkstra's algorithm with φ-weighted edges.
        """
        start = start.lower()
        end = end.lower()
        
        if start not in self.nodes or end not in self.nodes:
            return None
        
        # Dijkstra's algorithm
        distances = {start: 0.0}
        previous = {}
        pq = [(0.0, start)]
        visited = set()
        
        while pq:
            dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == end:
                # Reconstruct path
                path = []
                node = end
                while node in previous:
                    path.append(node)
                    node = previous[node]
                path.append(start)
                path.reverse()
                
                return GeodesicPath(nodes=path, total_weight=dist)
            
            if len(visited) > max_length * 10:  # Limit search
                break
            
            node = self.nodes[current]
            for neighbor, weight in node.neighbors.items():
                if neighbor in visited:
                    continue
                
                # Distance is inverse of weight (stronger connection = shorter distance)
                edge_dist = 1.0 / (weight + 0.1)
                new_dist = dist + edge_dist
                
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
        
        return None
    
    def explore_neighborhood(self, concept: str, depth: int = 2, max_per_level: int = 3) -> List[str]:
        """
        Explore concepts reachable from a starting concept.
        
        Returns concepts in order of discovery (BFS with φ-weighting).
        """
        concept = concept.lower()
        if concept not in self.nodes:
            return [concept]
        
        visited = {concept}
        result = [concept]
        frontier = [concept]
        
        for _ in range(depth):
            next_frontier = []
            
            for current in frontier:
                if current not in self.nodes:
                    continue
                
                node = self.nodes[current]
                
                # Sort neighbors by weight (strongest connections first)
                sorted_neighbors = sorted(
                    node.neighbors.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                count = 0
                for neighbor, weight in sorted_neighbors:
                    if neighbor not in visited and count < max_per_level:
                        visited.add(neighbor)
                        result.append(neighbor)
                        next_frontier.append(neighbor)
                        count += 1
            
            frontier = next_frontier
        
        return result
    
    def generate_sentence(self, subject: str, verb: Optional[str] = None, obj: Optional[str] = None) -> str:
        """
        Generate a single sentence from concept components.
        
        Uses the geometric knowledge to find appropriate verbs/objects
        if not provided.
        """
        subject = subject.lower()
        
        if subject not in self.nodes:
            return f"{subject.title()} exists."
        
        node = self.nodes[subject]
        
        # Find verb if not provided
        if not verb and node.actions:
            # Pick most common action
            verb = max(node.actions.items(), key=lambda x: x[1])[0]
        
        # Find object if not provided
        if not obj and node.targets:
            # Pick most common target
            obj = max(node.targets.items(), key=lambda x: x[1])[0]
        
        # Build sentence based on what we have
        if verb and obj:
            return f"{subject.title()} {verb} {obj}."
        elif verb:
            return f"{subject.title()} {verb}."
        elif obj:
            return f"{subject.title()} involves {obj}."
        else:
            # Describe based on φ-direction
            if node.phi_direction > 0.3:
                return f"{subject.title()} is an initiator."
            elif node.phi_direction < -0.3:
                return f"{subject.title()} is affected by others."
            else:
                return f"{subject.title()} plays a mediating role."
    
    def generate_about(self, concept: str, num_sentences: int = 3) -> str:
        """
        Generate free-form text about a concept by exploring its neighborhood.
        
        This is the main entry point for geodesic generation.
        """
        concept = concept.lower()
        
        if concept not in self.nodes:
            return f"I don't have information about {concept}."
        
        # Explore neighborhood
        path = self.explore_neighborhood(concept, depth=num_sentences + 1, max_per_level=2)
        
        sentences = []
        covered_verbs = set()
        covered_relations = set()  # Track (subject, object) pairs to avoid repetition
        
        for i, node_name in enumerate(path):
            if len(sentences) >= num_sentences:
                break
            if node_name not in self.nodes:
                continue
            
            node = self.nodes[node_name]
            
            if i == 0:
                # First sentence: describe the main concept
                sentence = self._describe_concept(node_name, node, covered_relations)
            else:
                # Subsequent sentences: relate to previous or describe
                prev_name = path[i-1] if i > 0 else concept
                sentence = self._relate_concepts(prev_name, node_name, covered_verbs, covered_relations)
            
            if sentence and sentence not in sentences:
                sentences.append(sentence)
        
        return " ".join(sentences)
    
    # Sentence templates for more natural prose
    # {action} = present tense (examines), {action_base} = infinitive (examine)
    TEMPLATES = {
        'action_target': [
            "{name} {action} the {target}.",
            "{name} is known to {action_base} {target}.",
            "The {role} {name} {action} {target}.",
        ],
        'action_only': [
            "{name} {action} with skill.",
            "{name} is known for {action_gerund}.",
        ],
        'role_initiator': [
            "{name} is a key figure who {action}.",
            "As a protagonist, {name} {action}.",
            "{name} takes an active role.",
        ],
        'role_receiver': [
            "{name} is central to the narrative.",
            "{name} plays an important part.",
        ],
        'relation': [
            "{name} is connected to {neighbor}.",
            "{name} and {neighbor} are linked.",
        ],
    }
    
    def _to_gerund(self, verb: str) -> str:
        """Convert verb to gerund form."""
        if verb.endswith('e') and not verb.endswith('ee'):
            return verb[:-1] + 'ing'
        elif verb.endswith('ing'):
            return verb
        else:
            return verb + 'ing'
    
    # Irregular verb mappings (past -> base)
    IRREGULAR_VERBS = {
        'was': 'be', 'were': 'be', 'been': 'be',
        'had': 'have', 'has': 'have',
        'did': 'do', 'does': 'do',
        'went': 'go', 'gone': 'go', 'goes': 'go',
        'came': 'come', 'comes': 'come',
        'made': 'make', 'makes': 'make',
        'took': 'take', 'takes': 'take', 'taken': 'take',
        'got': 'get', 'gets': 'get',
        'knew': 'know', 'knows': 'know', 'known': 'know',
        'thought': 'think', 'thinks': 'think',
        'saw': 'see', 'sees': 'see', 'seen': 'see',
        'said': 'say', 'says': 'say',
        'told': 'tell', 'tells': 'tell',
        'found': 'find', 'finds': 'find',
        'gave': 'give', 'gives': 'give', 'given': 'give',
        'left': 'leave', 'leaves': 'leave',
        'became': 'become', 'becomes': 'become',
        'kept': 'keep', 'keeps': 'keep',
        'brought': 'bring', 'brings': 'bring',
        'began': 'begin', 'begins': 'begin', 'begun': 'begin',
        'wrote': 'write', 'writes': 'write', 'written': 'write',
        'ran': 'run', 'runs': 'run',
        'read': 'read', 'reads': 'read',
        'spoke': 'speak', 'speaks': 'speak', 'spoken': 'speak',
        'stood': 'stand', 'stands': 'stand',
        'understood': 'understand', 'understands': 'understand',
        'held': 'hold', 'holds': 'hold',
        'heard': 'hear', 'hears': 'hear',
        'met': 'meet', 'meets': 'meet',
        'set': 'set', 'sets': 'set',
        'sat': 'sit', 'sits': 'sit',
        'led': 'lead', 'leads': 'lead',
        'felt': 'feel', 'feels': 'feel',
        'fell': 'fall', 'falls': 'fall', 'fallen': 'fall',
        'sent': 'send', 'sends': 'send',
        'built': 'build', 'builds': 'build',
        'lost': 'lose', 'loses': 'lose',
        'paid': 'pay', 'pays': 'pay',
        'spent': 'spend', 'spends': 'spend',
        'caught': 'catch', 'catches': 'catch',
        'taught': 'teach', 'teaches': 'teach',
        'bought': 'buy', 'buys': 'buy',
        'fought': 'fight', 'fights': 'fight',
        'sought': 'seek', 'seeks': 'seek',
        'won': 'win', 'wins': 'win',
        'drew': 'draw', 'draws': 'draw', 'drawn': 'draw',
        'grew': 'grow', 'grows': 'grow', 'grown': 'grow',
        'threw': 'throw', 'throws': 'throw', 'thrown': 'throw',
        'flew': 'fly', 'flies': 'fly', 'flown': 'fly',
        'drove': 'drive', 'drives': 'drive', 'driven': 'drive',
        'rose': 'rise', 'rises': 'rise', 'risen': 'rise',
        'chose': 'choose', 'chooses': 'choose', 'chosen': 'choose',
        'broke': 'break', 'breaks': 'break', 'broken': 'break',
        'wore': 'wear', 'wears': 'wear', 'worn': 'wear',
        'ate': 'eat', 'eats': 'eat', 'eaten': 'eat',
        'drank': 'drink', 'drinks': 'drink', 'drunk': 'drink',
        'sang': 'sing', 'sings': 'sing', 'sung': 'sing',
        'swam': 'swim', 'swims': 'swim', 'swum': 'swim',
    }
    
    def _to_base_form(self, verb: str) -> str:
        """Convert verb to base form (infinitive)."""
        v = verb.lower()
        
        # Check irregular verbs first
        if v in self.IRREGULAR_VERBS:
            return self.IRREGULAR_VERBS[v]
        
        # Handle common past tense patterns
        if v.endswith('ed'):
            if v.endswith('ied'):
                return v[:-3] + 'y'  # studied -> study
            elif v.endswith('eed'):
                return v[:-2]  # agreed -> agree
            elif len(v) > 4 and v[-4] == v[-3]:  # doubled consonant
                return v[:-3]  # stopped -> stop
            else:
                base = v[:-2] if len(v) > 2 else v
                # Check if we need to add back 'e' (loved -> love)
                if base and base[-1] in 'bcdfghjklmnpqrstvwxyz':
                    # Common patterns that need 'e' added back
                    if base.endswith(('lov', 'examin', 'observ', 'deduc', 'describ',
                                     'explor', 'transform', 'provid', 'includ',
                                     'creat', 'produc', 'reduc', 'introduc')):
                        return base + 'e'
                return base if base else v
        
        # Handle -ing forms
        if v.endswith('ing') and len(v) > 4:
            base = v[:-3]
            # Check for doubled consonant
            if len(base) > 1 and base[-1] == base[-2]:
                return base[:-1]  # running -> run
            # Check if needs 'e'
            if base.endswith(('mak', 'tak', 'giv', 'hav', 'com', 'writ', 'driv')):
                return base + 'e'
            return base
        
        # Handle -s/-es forms (third person singular)
        if v.endswith('ies') and len(v) > 4:
            return v[:-3] + 'y'  # studies -> study
        if v.endswith('es') and len(v) > 3:
            base = v[:-2]
            if base.endswith(('ch', 'sh', 'ss', 'x', 'z')):
                return base  # watches -> watch
            return v[:-1]  # explores -> explore
        if v.endswith('s') and len(v) > 2 and not v.endswith('ss'):
            return v[:-1]  # walks -> walk
        
        return v
    
    def _to_present_tense(self, verb: str) -> str:
        """Convert verb to third-person present tense."""
        base = self._to_base_form(verb)
        
        # Handle special cases
        if base == 'be':
            return 'is'
        if base == 'have':
            return 'has'
        if base == 'do':
            return 'does'
        if base == 'go':
            return 'goes'
        
        if base.endswith('y') and len(base) > 1 and base[-2] not in 'aeiou':
            return base[:-1] + 'ies'  # study -> studies
        elif base.endswith(('s', 'sh', 'ch', 'x', 'z', 'o')):
            return base + 'es'  # watch -> watches
        else:
            return base + 's'  # walk -> walks
    
    def _infer_role(self, node: ConceptNode) -> str:
        """Infer a role word for the concept."""
        if node.phi_direction > 0.5:
            return "protagonist"
        elif node.phi_direction > 0.2:
            return "character"
        elif node.phi_direction < -0.3:
            return "subject"
        else:
            return "figure"
    
    def _describe_concept(self, name: str, node: ConceptNode, covered_relations: Set[Tuple[str, str]] = None) -> str:
        """Generate a descriptive sentence about a concept."""
        import random
        
        if covered_relations is None:
            covered_relations = set()
        
        # Action-based description with targets
        if node.actions and node.targets:
            top_action = max(node.actions.items(), key=lambda x: x[1])[0]
            top_target = max(node.targets.items(), key=lambda x: x[1])[0]
            relation = (name, top_target)
            if relation not in covered_relations:
                covered_relations.add(relation)
                role = self._infer_role(node)
                base_action = self._to_base_form(top_action)
                template = random.choice(self.TEMPLATES['action_target'])
                base_action = self._to_base_form(top_action)
                return template.format(
                    name=name.title(), 
                    action=self._to_present_tense(top_action),
                    action_base=base_action,
                    target=top_target,
                    role=role
                )
        
        # Action-based description
        if node.actions:
            top_actions = sorted(node.actions.items(), key=lambda x: x[1], reverse=True)[:2]
            action = top_actions[0][0]
            base_action = self._to_base_form(action)
            template = random.choice(self.TEMPLATES['action_only'])
            return template.format(
                name=name.title(),
                action=self._to_present_tense(action),
                action_gerund=self._to_gerund(base_action)
            )
        
        # Role-based description using φ-direction
        if node.phi_direction > 0.3:
            if node.neighbors:
                top_neighbor = max(node.neighbors.items(), key=lambda x: x[1])[0]
                return f"{name.title()} is a key figure connected to {top_neighbor}."
            template = random.choice(self.TEMPLATES['role_initiator'])
            action = list(node.actions.keys())[0] if node.actions else "acts"
            return template.format(name=name.title(), action=action)
        elif node.phi_direction < -0.3:
            template = random.choice(self.TEMPLATES['role_receiver'])
            return template.format(name=name.title())
        
        # Neighbor-based description
        if node.neighbors:
            top_neighbor = max(node.neighbors.items(), key=lambda x: x[1])[0]
            template = random.choice(self.TEMPLATES['relation'])
            return template.format(name=name.title(), neighbor=top_neighbor)
        
        return f"{name.title()} appears in the narrative."
    
    def _relate_concepts(self, prev: str, current: str, covered_verbs: Set[str], covered_relations: Set[Tuple[str, str]] = None) -> str:
        """Generate a sentence relating two concepts."""
        if covered_relations is None:
            covered_relations = set()
        
        prev = prev.lower()
        current = current.lower()
        
        if current not in self.nodes:
            return ""
        
        node = self.nodes[current]
        
        # Check if prev is a source of current
        if prev in node.sources:
            # Find the verb that connects them
            if self.knowledge:
                for frame in self.knowledge.frames:
                    init = (frame.initiator or '').lower()
                    med = (frame.mediator or '').lower()
                    recv = (frame.receiver or '').lower()
                    
                    if init == prev and recv == current and med and med not in covered_verbs:
                        covered_verbs.add(med)
                        return f"{prev.title()} {med} {current}."
        
        # Check if current has actions involving prev
        if prev in node.targets:
            for action in node.actions:
                if action not in covered_verbs:
                    covered_verbs.add(action)
                    return f"{current.title()} {action} {prev}."
        
        # Default: describe current
        return self._describe_concept(current, node)
    
    def generate_comparison(self, concept_a: str, concept_b: str) -> str:
        """Generate text comparing two concepts."""
        concept_a = concept_a.lower()
        concept_b = concept_b.lower()
        
        if concept_a not in self.nodes or concept_b not in self.nodes:
            return f"Cannot compare {concept_a} and {concept_b}."
        
        node_a = self.nodes[concept_a]
        node_b = self.nodes[concept_b]
        
        sentences = []
        
        # Compare φ-directions
        if node_a.phi_direction > 0.3 and node_b.phi_direction > 0.3:
            sentences.append(f"Both {concept_a.title()} and {concept_b.title()} are initiators.")
        elif node_a.phi_direction < -0.3 and node_b.phi_direction < -0.3:
            sentences.append(f"Both {concept_a.title()} and {concept_b.title()} are receivers.")
        elif abs(node_a.phi_direction - node_b.phi_direction) > 0.5:
            if node_a.phi_direction > node_b.phi_direction:
                sentences.append(f"{concept_a.title()} is more active, while {concept_b.title()} is more passive.")
            else:
                sentences.append(f"{concept_b.title()} is more active, while {concept_a.title()} is more passive.")
        
        # Compare actions
        actions_a = set(node_a.actions.keys())
        actions_b = set(node_b.actions.keys())
        shared_actions = actions_a & actions_b
        
        if shared_actions:
            action = list(shared_actions)[0]
            sentences.append(f"Both {action}.")
        
        # Find path between them
        path = self.find_geodesic(concept_a, concept_b)
        if path and len(path.nodes) > 2:
            middle = path.nodes[len(path.nodes) // 2]
            sentences.append(f"They are connected through {middle}.")
        
        return " ".join(sentences) if sentences else f"{concept_a.title()} and {concept_b.title()} are different concepts."
    
    def generate_story(self, concepts: List[str], max_sentences: int = 5) -> str:
        """
        Generate a short narrative connecting multiple concepts.
        
        Finds geodesic paths between concepts and generates
        sentences along the way.
        """
        if not concepts:
            return ""
        
        concepts = [c.lower() for c in concepts if c.lower() in self.nodes]
        if not concepts:
            return "No known concepts provided."
        
        sentences = []
        covered = set()
        
        # Start with first concept
        current = concepts[0]
        sentences.append(self._describe_concept(current, self.nodes[current]))
        covered.add(current)
        
        # Navigate to each subsequent concept
        for target in concepts[1:]:
            if len(sentences) >= max_sentences:
                break
            
            path = self.find_geodesic(current, target)
            if path:
                # Generate sentences along path (skip endpoints we've covered)
                for node_name in path.nodes[1:]:
                    if len(sentences) >= max_sentences:
                        break
                    if node_name not in covered:
                        sentence = self._relate_concepts(current, node_name, set())
                        if sentence:
                            sentences.append(sentence)
                        covered.add(node_name)
                        current = node_name
            else:
                # No path found, just describe target
                if target not in covered:
                    sentences.append(self._describe_concept(target, self.nodes[target]))
                    covered.add(target)
                    current = target
        
        return " ".join(sentences)


def demo():
    """Demonstrate geodesic generation."""
    print("=" * 70)
    print("GEODESIC GENERATOR DEMO")
    print("Free-form text via concept space navigation")
    print("=" * 70)
    
    # Create with sample knowledge
    from .geometric import GeometricKnowledge
    
    knowledge = GeometricKnowledge()
    
    # Add sample sentences
    sentences = [
        "Holmes examines the evidence carefully.",
        "Holmes deduces the identity of the criminal.",
        "Watson assists Holmes in the investigation.",
        "Watson writes about their adventures.",
        "Moriarty challenges Holmes intellectually.",
        "Moriarty plans elaborate schemes.",
        "Lestrade arrests the criminals.",
        "Lestrade consults Holmes on difficult cases.",
        "The evidence reveals the truth.",
        "The criminal escapes from custody.",
        "Darcy loves Elizabeth deeply.",
        "Elizabeth challenges Darcy's pride.",
        "Jane admires Bingley's kindness.",
        "Bingley courts Jane at the ball.",
        "Wickham deceives the Bennet family.",
    ]
    
    for s in sentences:
        knowledge.learn(s, "demo")
    
    # Create generator
    gen = GeodesicGenerator(knowledge)
    
    print(f"\nBuilt graph with {len(gen.nodes)} concept nodes")
    print()
    
    # Test neighborhood exploration
    print("-" * 70)
    print("NEIGHBORHOOD EXPLORATION")
    print("-" * 70)
    
    for concept in ["holmes", "darcy", "watson"]:
        neighbors = gen.explore_neighborhood(concept, depth=2)
        print(f"\n{concept.title()}'s neighborhood: {neighbors}")
    
    # Test geodesic paths
    print("\n" + "-" * 70)
    print("GEODESIC PATHS")
    print("-" * 70)
    
    pairs = [
        ("holmes", "watson"),
        ("holmes", "moriarty"),
        ("darcy", "elizabeth"),
        ("holmes", "darcy"),
    ]
    
    for a, b in pairs:
        path = gen.find_geodesic(a, b)
        if path:
            print(f"\n{a} → {b}: {' → '.join(path.nodes)} (weight: {path.total_weight:.2f})")
        else:
            print(f"\n{a} → {b}: No path found")
    
    # Test free-form generation
    print("\n" + "-" * 70)
    print("FREE-FORM GENERATION")
    print("-" * 70)
    
    for concept in ["holmes", "darcy", "watson", "moriarty"]:
        print(f"\nAbout {concept.title()}:")
        text = gen.generate_about(concept, num_sentences=3)
        print(f"  {text}")
    
    # Test comparison
    print("\n" + "-" * 70)
    print("CONCEPT COMPARISON")
    print("-" * 70)
    
    comparisons = [
        ("holmes", "watson"),
        ("darcy", "elizabeth"),
        ("holmes", "moriarty"),
    ]
    
    for a, b in comparisons:
        print(f"\nComparing {a.title()} and {b.title()}:")
        text = gen.generate_comparison(a, b)
        print(f"  {text}")
    
    # Test story generation
    print("\n" + "-" * 70)
    print("STORY GENERATION")
    print("-" * 70)
    
    concept_lists = [
        ["holmes", "evidence", "criminal"],
        ["darcy", "elizabeth", "jane"],
        ["watson", "holmes", "moriarty"],
    ]
    
    for concepts in concept_lists:
        print(f"\nStory about {concepts}:")
        text = gen.generate_story(concepts, max_sentences=4)
        print(f"  {text}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    demo()
