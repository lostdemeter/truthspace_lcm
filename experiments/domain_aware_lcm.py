#!/usr/bin/env python3
"""
Domain-Aware GeometricLCM

Key insight: Holographic interference showed that all protagonists look the same
structurally. We need DOMAIN TRACKING to separate topics while maintaining
geometric overlap.

The domain dimension is like the 't' coordinate on the zeta critical line:
  σ = 0.5 (shared structure)
  t = domain frequency (which story/topic)

This allows:
  - Cross-domain queries: "Who examines?" → Holmes, Hamlet (both investigate)
  - Domain-specific queries: "Who examines in Sherlock Holmes?" → Holmes
  - Domain transfer: "What would Alice do in a mystery?" → examine (like Holmes)

Author: Lesley Gushurst
License: GPLv3
"""

import os
import sys
import re
import random
import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.morphological_quaternion import MorphoQuaternion, MorphologicalTransformer

PHI = 1.618034
PI = math.pi


# Domain definitions with keywords for detection
DOMAINS = {
    'sherlock': {
        'keywords': {'holmes', 'watson', 'detective', 'lestrade', 'moriarty', 
                    'mycroft', 'hudson', 'baker', 'street', 'elementary'},
        'genre': 'mystery',
        't_value': 0.0,  # Zeta t-coordinate
    },
    'alice': {
        'keywords': {'alice', 'rabbit', 'queen', 'cheshire', 'hatter', 'dormouse',
                    'caterpillar', 'wonderland', 'tea', 'croquet'},
        'genre': 'fantasy',
        't_value': 1.0,
    },
    'pride': {
        'keywords': {'darcy', 'elizabeth', 'bennet', 'bingley', 'wickham', 'jane',
                    'lydia', 'catherine', 'longbourn', 'pemberley'},
        'genre': 'romance',
        't_value': 2.0,
    },
    'gatsby': {
        'keywords': {'gatsby', 'nick', 'daisy', 'tom', 'wilson', 'myrtle',
                    'buchanan', 'carraway', 'green', 'light'},
        'genre': 'tragedy',
        't_value': 3.0,
    },
    'hamlet': {
        'keywords': {'hamlet', 'claudius', 'gertrude', 'ophelia', 'polonius',
                    'laertes', 'fortinbras', 'horatio', 'ghost', 'elsinore'},
        'genre': 'tragedy',
        't_value': 4.0,
    },
}


@dataclass
class DomainConcept:
    """A concept with domain tracking."""
    word: str
    
    # Domain membership (can belong to multiple)
    domains: Counter = field(default_factory=Counter)
    
    # Role counts per domain
    actor_by_domain: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    action_by_domain: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    target_by_domain: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Global counts
    actor_count: int = 0
    action_count: int = 0
    target_count: int = 0
    
    # Relationships per domain
    actions_by_domain: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    targets_by_domain: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    
    # φ-direction (from symmetric understanding)
    phi_direction: float = 0.0
    
    @property
    def primary_domain(self) -> Optional[str]:
        """Get the most common domain for this concept."""
        if not self.domains:
            return None
        return self.domains.most_common(1)[0][0]
    
    @property
    def is_cross_domain(self) -> bool:
        """Does this concept appear in multiple domains?"""
        return len(self.domains) > 1
    
    def domain_strength(self, domain: str) -> float:
        """How strongly is this concept associated with a domain?"""
        total = sum(self.domains.values())
        if total == 0:
            return 0.0
        return self.domains.get(domain, 0) / total


@dataclass
class DomainFrame:
    """A frame with domain context."""
    actor: str
    action: str
    target: Optional[str]
    domain: str
    source_sentence: str = ""


class DomainAwareIngester:
    """
    Ingester that tracks domain membership for each concept.
    
    This enables:
    1. Domain-specific queries ("Who in Sherlock Holmes?")
    2. Cross-domain queries ("Who examines across all stories?")
    3. Domain transfer ("What would X do in domain Y?")
    """
    
    def __init__(self):
        self.concepts: Dict[str, DomainConcept] = {}
        self.frames: List[DomainFrame] = []
        self.sentences_by_domain: Dict[str, List[str]] = defaultdict(list)
        
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
        """Detect which domain a piece of text belongs to."""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        best_domain = 'unknown'
        best_score = 0
        
        for domain, info in DOMAINS.items():
            overlap = len(words & info['keywords'])
            if overlap > best_score:
                best_score = overlap
                best_domain = domain
        
        return best_domain
    
    def _get_or_create_concept(self, word: str) -> DomainConcept:
        word_lower = word.lower()
        if word_lower not in self.concepts:
            self.concepts[word_lower] = DomainConcept(word=word_lower)
        return self.concepts[word_lower]
    
    def _extract_frame(self, sentence: str, domain: str) -> Optional[DomainFrame]:
        """Extract frame with domain context."""
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        content = [t for t in tokens if t not in self.function_words and len(t) > 2]
        
        if len(content) < 2:
            return None
        
        actor = content[0]
        action = content[1]
        target = content[2] if len(content) > 2 else None
        
        # Skip adverb targets
        if target and (target.endswith('ly') or len(target) <= 3):
            target = content[3] if len(content) > 3 else None
        
        return DomainFrame(
            actor=actor,
            action=action,
            target=target,
            domain=domain,
            source_sentence=sentence.strip(),
        )
    
    def ingest(self, text: str):
        """Ingest text with domain tracking."""
        # Split by paragraph/section (domains often change at paragraph boundaries)
        paragraphs = re.split(r'\n\s*\n', text)
        
        for para in paragraphs:
            # Detect domain for this paragraph
            domain = self._detect_domain(para)
            
            # Split into sentences
            sentences = re.split(r'[.!?]+', para)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Re-detect domain at sentence level for mixed paragraphs
                sent_domain = self._detect_domain(sentence)
                if sent_domain != 'unknown':
                    domain = sent_domain
                
                self.sentences_by_domain[domain].append(sentence)
                
                # Extract frame
                frame = self._extract_frame(sentence, domain)
                if frame:
                    self.frames.append(frame)
                    self._update_concepts(frame)
        
        # Compute φ-directions
        self._compute_directions()
    
    def _update_concepts(self, frame: DomainFrame):
        """Update concepts from a frame."""
        domain = frame.domain
        
        # Actor
        actor = self._get_or_create_concept(frame.actor)
        actor.domains[domain] += 1
        actor.actor_by_domain[domain] += 1
        actor.actor_count += 1
        actor.actions_by_domain[domain][frame.action] += 1
        
        # Action
        action = self._get_or_create_concept(frame.action)
        action.domains[domain] += 1
        action.action_by_domain[domain] += 1
        action.action_count += 1
        
        # Target
        if frame.target:
            target = self._get_or_create_concept(frame.target)
            target.domains[domain] += 1
            target.target_by_domain[domain] += 1
            target.target_count += 1
            
            actor.targets_by_domain[domain][frame.target] += 1
    
    def _compute_directions(self):
        """Compute φ-directions for all concepts."""
        for concept in self.concepts.values():
            entity_count = concept.actor_count + concept.target_count
            total = entity_count + concept.action_count
            if total > 0:
                concept.phi_direction = (entity_count - concept.action_count) / total
    
    def get_domain_entities(self, domain: str) -> List[str]:
        """Get entities that belong to a domain."""
        entities = []
        for name, concept in self.concepts.items():
            if domain in concept.domains and concept.actor_count > 0:
                entities.append(name)
        return entities
    
    def get_cross_domain_concepts(self) -> List[str]:
        """Get concepts that appear in multiple domains."""
        return [name for name, c in self.concepts.items() if c.is_cross_domain]
    
    def find_similar_in_domain(self, entity: str, target_domain: str) -> List[Tuple[str, float]]:
        """
        Find entities in target_domain that are similar to entity.
        
        This enables domain transfer: "What would Alice do in a mystery?"
        """
        if entity not in self.concepts:
            return []
        
        source = self.concepts[entity]
        source_actions = set()
        for domain_actions in source.actions_by_domain.values():
            source_actions.update(domain_actions.keys())
        
        candidates = []
        for name, concept in self.concepts.items():
            if name == entity:
                continue
            if target_domain not in concept.domains:
                continue
            if concept.actor_count == 0:
                continue
            
            # Similarity based on shared actions
            target_actions = set(concept.actions_by_domain.get(target_domain, {}).keys())
            shared = len(source_actions & target_actions)
            
            # Also consider φ-direction similarity
            dir_sim = 1.0 - abs(source.phi_direction - concept.phi_direction)
            
            score = shared + dir_sim
            if score > 0:
                candidates.append((name, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def get_statistics(self) -> Dict:
        """Get ingestion statistics."""
        return {
            'total_concepts': len(self.concepts),
            'total_frames': len(self.frames),
            'domains': {
                domain: len(sentences) 
                for domain, sentences in self.sentences_by_domain.items()
            },
            'cross_domain_concepts': len(self.get_cross_domain_concepts()),
        }


@dataclass
class QuaternionSettings:
    style: str = 'neutral'
    certainty: str = 'neutral'
    depth: str = 'moderate'
    person: str = '3rd'
    number: str = 'singular'
    tense: str = 'present'
    aspect: str = 'simple'


class DomainAwareLCM:
    """
    GeometricLCM with domain awareness.
    
    Enables:
    1. Domain-specific generation: "Tell me about Holmes" → Sherlock domain
    2. Cross-domain queries: "Who investigates?" → Holmes, Hamlet
    3. Domain transfer: "What would Alice do as a detective?"
    """
    
    def __init__(self):
        self.ingester = DomainAwareIngester()
        self.morpho = MorphologicalTransformer()
        self.settings = QuaternionSettings()
        self.current_domain: Optional[str] = None  # Domain filter
        
        self.action_implications = {
            'examine': {'quality': 'analytical', 'domain': 'investigation'},
            'observe': {'quality': 'perceptive', 'domain': 'investigation'},
            'study': {'quality': 'methodical', 'domain': 'investigation'},
            'deduce': {'quality': 'brilliant', 'domain': 'reasoning'},
            'write': {'quality': 'diligent', 'domain': 'documentation'},
            'watch': {'quality': 'vigilant', 'domain': 'observation'},
            'look': {'quality': 'attentive', 'domain': 'observation'},
            'fall': {'quality': 'vulnerable', 'domain': 'transformation'},
            'grow': {'quality': 'dynamic', 'domain': 'transformation'},
            'love': {'quality': 'devoted', 'domain': 'emotion'},
            'kill': {'quality': 'decisive', 'domain': 'action'},
            'plot': {'quality': 'cunning', 'domain': 'scheming'},
            'pursue': {'quality': 'determined', 'domain': 'action'},
        }
    
    def ingest(self, text: str):
        self.ingester.ingest(text)
    
    def set_domain(self, domain: str = None):
        """Set domain filter for queries."""
        self.current_domain = domain
    
    def set_style(self, style: str = None, certainty: str = None, depth: str = None):
        if style:
            self.settings.style = style
        if certainty:
            self.settings.certainty = certainty
        if depth:
            self.settings.depth = depth
    
    def set_morphology(self, person: str = None, number: str = None,
                       tense: str = None, aspect: str = None):
        if person:
            self.settings.person = person
        if number:
            self.settings.number = number
        if tense:
            self.settings.tense = tense
        if aspect:
            self.settings.aspect = aspect
    
    def _get_morpho_quaternion(self) -> MorphoQuaternion:
        person_map = {'1st': -1, '2nd': 0, '3rd': 1}
        number_map = {'singular': -1, 'plural': 1}
        tense_map = {'past': -1, 'present': 0, 'future': 1}
        aspect_map = {'simple': -1, 'perfect': 0, 'progressive': 1}
        
        return MorphoQuaternion(
            x=person_map.get(self.settings.person, 1),
            y=number_map.get(self.settings.number, -1),
            z=tense_map.get(self.settings.tense, 0),
            w=aspect_map.get(self.settings.aspect, -1),
        )
    
    def _conjugate(self, verb: str) -> str:
        base = self.morpho._get_base(verb)
        q3 = self._get_morpho_quaternion()
        return self.morpho.transform(base, q3)
    
    def _get_certainty_opener(self) -> str:
        if self.settings.certainty == 'definitive':
            return random.choice(['Certainly,', 'Without question,', 'Undoubtedly,']) + ' '
        elif self.settings.certainty == 'hedged':
            return random.choice(['Perhaps', 'It seems that', 'Arguably,']) + ' '
        return ''
    
    def _format_target(self, target: str) -> Optional[str]:
        if not target:
            return None
        
        bad_targets = {'tall', 'small', 'confused', 'scared', 'angrily', 'intently',
                      'gracefully', 'wildly', 'slowly', 'quickly', 'carefully',
                      'methodically', 'proudly', 'completely', 'mysteriously',
                      'him', 'her', 'them', 'it', 'against', 'through', 'afar',
                      'immediately', 'eventually', 'finally', 'suddenly', 'briefly',
                      'constantly', 'desperately', 'triumphantly', 'secretly'}
        
        if target in bad_targets or target.endswith('ly'):
            return None
        
        common_nouns = {'evidence', 'room', 'journal', 'newspaper', 'garden',
                       'building', 'window', 'scene', 'tea', 'hole', 'footprints',
                       'witnesses', 'doorway', 'rabbit', 'villain', 'ball', 'party'}
        
        if target in common_nouns:
            return f"the {target}"
        
        return target.title()
    
    def _get_action_quality(self, action: str) -> str:
        base = self.morpho._get_base(action)
        if base in self.action_implications:
            return self.action_implications[base]['quality']
        return 'notable'
    
    def generate(self, seed: str = None, domain: str = None, num_sentences: int = 1) -> str:
        """Generate text, optionally constrained to a domain."""
        domain = domain or self.current_domain
        sentences = []
        current_seed = seed.lower() if seed else None
        
        # If no seed, pick from domain
        if not current_seed:
            if domain:
                entities = self.ingester.get_domain_entities(domain)
            else:
                entities = [n for n, c in self.ingester.concepts.items() if c.actor_count > 0]
            if entities:
                current_seed = random.choice(entities)
        
        for _ in range(num_sentences):
            if not current_seed or current_seed not in self.ingester.concepts:
                break
            
            concept = self.ingester.concepts[current_seed]
            
            # Get actions (domain-filtered if specified)
            if domain and domain in concept.actions_by_domain:
                actions = concept.actions_by_domain[domain]
            else:
                # Merge all domain actions
                actions = Counter()
                for d_actions in concept.actions_by_domain.values():
                    actions.update(d_actions)
            
            if not actions:
                break
            
            # Filter to real verbs (not nouns that slipped through)
            real_actions = Counter()
            for act, count in actions.items():
                # Check if this word is actually used as an action
                if act in self.ingester.concepts:
                    act_concept = self.ingester.concepts[act]
                    if act_concept.action_count > 0:
                        real_actions[act] = count
            
            if not real_actions:
                # Fallback to original actions
                real_actions = actions
            
            action = real_actions.most_common(1)[0][0]
            
            # Get target (domain-filtered)
            if domain and domain in concept.targets_by_domain:
                targets = concept.targets_by_domain[domain]
            else:
                targets = Counter()
                for d_targets in concept.targets_by_domain.values():
                    targets.update(d_targets)
            
            target = None
            if targets:
                for t, _ in targets.most_common(5):
                    # Skip if target is same as seed or action
                    if t == current_seed or t == action:
                        continue
                    formatted = self._format_target(t)
                    if formatted:
                        target = t
                        break
            
            # Conjugate and format
            verb = self._conjugate(action)
            opener = self._get_certainty_opener()
            target_str = self._format_target(target)
            
            if self.settings.style == 'literary':
                quality = self._get_action_quality(action)
                if target_str:
                    sentence = f"{opener}{current_seed.title()}, a {quality} character, {verb} {target_str}."
                else:
                    sentence = f"{opener}{current_seed.title()} demonstrates {quality} character."
            else:
                if target_str:
                    sentence = f"{opener}{current_seed.title()} {verb} {target_str}.".strip()
                else:
                    sentence = f"{opener}{current_seed.title()} {verb}.".strip()
            
            sentences.append(sentence)
            
            # Chain to next entity (not action)
            next_seed = None
            if target and target in self.ingester.concepts:
                c = self.ingester.concepts[target]
                if c.actor_count > 0:
                    next_seed = target
            
            # If no good target, try to find another entity in same domain
            if not next_seed and domain:
                domain_entities = self.ingester.get_domain_entities(domain)
                other_entities = [e for e in domain_entities if e != current_seed]
                if other_entities:
                    next_seed = random.choice(other_entities)
            
            current_seed = next_seed
        
        return " ".join(sentences) if sentences else "I don't have enough information."
    
    def ask(self, question: str) -> str:
        """Answer a question with domain awareness."""
        question_lower = question.lower().strip().rstrip('?')
        
        # Detect domain from question
        domain = self.ingester._detect_domain(question)
        if domain == 'unknown':
            domain = self.current_domain
        
        # Handle different question types
        if 'who is' in question_lower:
            match = re.search(r'who\s+is\s+(\w+)', question_lower)
            if match:
                return self._describe_entity(match.group(1), domain)
        
        elif 'what does' in question_lower and 'do' in question_lower:
            match = re.search(r'what\s+does\s+(\w+)\s+do', question_lower)
            if match:
                return self._describe_actions(match.group(1), domain)
        
        elif 'who' in question_lower and any(w in question_lower for w in ['examine', 'watch', 'kill', 'love', 'investigate']):
            # Cross-domain action query
            action_match = re.search(r'who\s+(\w+)', question_lower)
            if action_match:
                return self._who_does_action(action_match.group(1), domain)
        
        elif 'would' in question_lower and 'do' in question_lower:
            # Domain transfer query
            match = re.search(r'what\s+would\s+(\w+)\s+do.*?(\w+)\s*\??$', question_lower)
            if match:
                entity = match.group(1)
                target_domain = self.ingester._detect_domain(match.group(2))
                return self._domain_transfer(entity, target_domain)
        
        elif 'compare' in question_lower:
            # Cross-domain comparison
            entities = re.findall(r'\b([A-Z][a-z]+)\b', question)
            if len(entities) >= 2:
                return self._compare_entities(entities[0].lower(), entities[1].lower())
        
        # Default: describe entity
        words = re.findall(r'\b\w+\b', question_lower)
        for word in words:
            if word in self.ingester.concepts:
                concept = self.ingester.concepts[word]
                if concept.actor_count > 0:
                    return self._describe_entity(word, domain)
        
        return "I don't have enough information to answer that."
    
    def _describe_entity(self, entity: str, domain: str = None) -> str:
        """Describe an entity, optionally within a domain context."""
        entity = entity.lower()
        if entity not in self.ingester.concepts:
            return f"I don't have information about {entity.title()}."
        
        concept = self.ingester.concepts[entity]
        opener = self._get_certainty_opener()
        
        # Get domain info
        primary = concept.primary_domain
        domain_info = ""
        if primary and primary != 'unknown':
            genre = DOMAINS.get(primary, {}).get('genre', 'narrative')
            domain_info = f" in the {genre} genre"
        
        # Get actions (domain-filtered if specified)
        if domain and domain in concept.actions_by_domain:
            actions = list(concept.actions_by_domain[domain].keys())[:3]
        else:
            actions = []
            for d_actions in concept.actions_by_domain.values():
                actions.extend(d_actions.keys())
            actions = list(set(actions))[:3]
        
        if actions:
            verbs = [self._conjugate(a) for a in actions]
            if self.settings.style == 'literary':
                quality = self._get_action_quality(actions[0])
                return f"{opener}{entity.title()} is a {quality} character{domain_info} who {', '.join(verbs)}."
            else:
                return f"{opener}{entity.title()} {', '.join(verbs)}{domain_info}."
        
        return f"{opener}{entity.title()} appears{domain_info}."
    
    def _describe_actions(self, entity: str, domain: str = None) -> str:
        """Describe what an entity does."""
        entity = entity.lower()
        if entity not in self.ingester.concepts:
            return f"I don't have information about {entity.title()}."
        
        concept = self.ingester.concepts[entity]
        opener = self._get_certainty_opener()
        
        if domain and domain in concept.actions_by_domain:
            actions = list(concept.actions_by_domain[domain].keys())[:3]
        else:
            actions = []
            for d_actions in concept.actions_by_domain.values():
                actions.extend(d_actions.keys())
            actions = list(set(actions))[:3]
        
        if actions:
            verbs = [self._conjugate(a) for a in actions]
            return f"{opener}{entity.title()} {', '.join(verbs)}."
        
        return f"{entity.title()} doesn't have recorded actions."
    
    def _who_does_action(self, action: str, domain: str = None) -> str:
        """Find who performs an action, optionally in a domain."""
        base_action = self.morpho._get_base(action)
        opener = self._get_certainty_opener()
        
        performers = []
        for name, concept in self.ingester.concepts.items():
            if concept.actor_count == 0:
                continue
            
            # Check if this entity does this action
            for d, d_actions in concept.actions_by_domain.items():
                if domain and d != domain:
                    continue
                for a in d_actions:
                    if self.morpho._get_base(a) == base_action:
                        performers.append((name, d_actions[a], d))
                        break
        
        if not performers:
            return f"No one appears to {action} in the text."
        
        # Group by domain
        by_domain = defaultdict(list)
        for name, count, d in performers:
            by_domain[d].append(name)
        
        if len(by_domain) == 1:
            d = list(by_domain.keys())[0]
            names = [n.title() for n in by_domain[d]]
            genre = DOMAINS.get(d, {}).get('genre', 'the narrative')
            return f"{opener}{', '.join(names)} {self._conjugate(action)} in the {genre}."
        else:
            # Cross-domain response
            parts = []
            for d, names in by_domain.items():
                genre = DOMAINS.get(d, {}).get('genre', d)
                parts.append(f"{', '.join([n.title() for n in names])} ({genre})")
            return f"{opener}Across domains: {'; '.join(parts)} all {self._conjugate(action)}."
    
    def _domain_transfer(self, entity: str, target_domain: str) -> str:
        """What would entity do in a different domain?"""
        entity = entity.lower()
        if entity not in self.ingester.concepts:
            return f"I don't have information about {entity.title()}."
        
        opener = self._get_certainty_opener()
        
        # Find similar entities in target domain
        similar = self.ingester.find_similar_in_domain(entity, target_domain)
        
        if not similar:
            return f"I can't determine what {entity.title()} would do in that domain."
        
        # Get actions from similar entity
        similar_entity = similar[0][0]
        similar_concept = self.ingester.concepts[similar_entity]
        
        if target_domain in similar_concept.actions_by_domain:
            actions = list(similar_concept.actions_by_domain[target_domain].keys())[:2]
            if actions:
                verbs = [self._conjugate(a) for a in actions]
                genre = DOMAINS.get(target_domain, {}).get('genre', 'that domain')
                return f"{opener}In a {genre}, {entity.title()} would likely {' and '.join(verbs)}, similar to {similar_entity.title()}."
        
        return f"I can't determine what {entity.title()} would do in that domain."
    
    def _compare_entities(self, entity1: str, entity2: str) -> str:
        """Compare two entities across domains."""
        if entity1 not in self.ingester.concepts or entity2 not in self.ingester.concepts:
            return "I don't have enough information to compare these entities."
        
        c1 = self.ingester.concepts[entity1]
        c2 = self.ingester.concepts[entity2]
        
        opener = self._get_certainty_opener()
        
        # Get domains
        d1 = c1.primary_domain
        d2 = c2.primary_domain
        
        # Get shared actions
        actions1 = set()
        for d_actions in c1.actions_by_domain.values():
            actions1.update(self.morpho._get_base(a) for a in d_actions)
        
        actions2 = set()
        for d_actions in c2.actions_by_domain.values():
            actions2.update(self.morpho._get_base(a) for a in d_actions)
        
        shared = actions1 & actions2
        
        if shared:
            shared_verbs = [self._conjugate(a) for a in list(shared)[:2]]
            g1 = DOMAINS.get(d1, {}).get('genre', d1)
            g2 = DOMAINS.get(d2, {}).get('genre', d2)
            return f"{opener}Both {entity1.title()} ({g1}) and {entity2.title()} ({g2}) {' and '.join(shared_verbs)}. They share similar roles in their narratives."
        else:
            return f"{opener}{entity1.title()} and {entity2.title()} have different roles in their respective stories."


# Expanded corpus with clear domain markers
DOMAIN_CORPUS = """
# Sherlock Holmes
Holmes examined the evidence carefully. Watson watched from the doorway.
The detective studied the footprints in the mud. He noticed something unusual about the pattern.
Holmes said to Watson that the case was elementary. Watson replied that he did not understand.
The inspector arrived at the scene promptly. Lestrade questioned the witnesses thoroughly.
Holmes observed the room methodically. He found a crucial clue near the window.
Watson wrote in his journal diligently. The doctor recorded every detail with precision.
Holmes deduced the killer's identity brilliantly. He explained his reasoning to the amazed audience.
The criminal fled through the garden desperately. Holmes pursued him quickly through the night.
Watson called for help immediately. The police surrounded the building completely.
Holmes captured the villain triumphantly. Justice was served at last.
Moriarty plotted against Holmes secretly. The professor was a criminal mastermind.
Mrs Hudson prepared tea for the gentlemen. She worried about their dangerous adventures.

# Alice in Wonderland
Alice fell down the rabbit hole unexpectedly. She wondered where she was going.
The Queen shouted angrily at everyone. Alice felt confused and scared.
The Cheshire Cat smiled mysteriously at Alice. He disappeared slowly into thin air.
Alice grew very tall suddenly. She shrank very small moments later.
The Mad Hatter laughed wildly at the party. He poured more tea endlessly.
The White Rabbit hurried past anxiously. He checked his watch constantly.
The Caterpillar smoked his hookah thoughtfully. He asked Alice strange questions.
The Dormouse slept peacefully at the table. He woke briefly to tell stories.

# Pride and Prejudice
Darcy looked at Elizabeth proudly. She ignored him completely at first.
Elizabeth danced gracefully at the ball. Darcy watched her intently from afar.
Mr Bennet read his newspaper quietly. Mrs Bennet worried about her daughters constantly.
Jane smiled sweetly at everyone. Bingley fell in love immediately with her.
Wickham deceived Elizabeth cunningly. He told lies about Darcy's character.
Lady Catherine visited Longbourn unexpectedly. She demanded Elizabeth refuse Darcy.
Lydia eloped with Wickham foolishly. The scandal threatened the family's reputation.
Darcy saved the Bennet family secretly. He paid Wickham to marry Lydia.
Elizabeth realized her mistake gradually. She began to appreciate Darcy's true character.

# The Great Gatsby
Gatsby watched the green light longingly. He dreamed of Daisy constantly.
Nick observed the wealthy parties curiously. He narrated the events thoughtfully.
Daisy cried over Gatsby's beautiful shirts. She had married Tom for money.
Tom confronted Gatsby aggressively. He revealed Gatsby's criminal connections.
Myrtle died in the accident tragically. Gatsby's car struck her on the road.
Wilson shot Gatsby in the pool. He believed Gatsby had killed Myrtle.
Nick arranged Gatsby's funeral alone. Nobody else attended the service.

# Hamlet
Hamlet pondered existence deeply. He questioned whether to live or die.
The ghost appeared to Hamlet mysteriously. He revealed Claudius murdered his father.
Claudius poisoned King Hamlet treacherously. He married Gertrude immediately after.
Ophelia loved Hamlet devotedly. She went mad from grief eventually.
Polonius spied on Hamlet foolishly. Hamlet killed him behind the curtain.
Laertes sought revenge passionately. He challenged Hamlet to a duel.
Gertrude drank the poisoned wine accidentally. She died before Hamlet's eyes.
Hamlet killed Claudius finally. He avenged his father at last.
"""


def run_demo():
    """Demonstrate domain-aware GeometricLCM."""
    print("=" * 70)
    print("DOMAIN-AWARE GEOMETRIC LCM")
    print("=" * 70)
    print()
    print("Key insight: Domain tracking enables topic separation while")
    print("maintaining geometric overlap for cross-domain queries.")
    print()
    print("Domains detected:")
    for domain, info in DOMAINS.items():
        print(f"  {domain}: {info['genre']} (t = {info['t_value']})")
    print()
    
    # Create model
    model = DomainAwareLCM()
    model.ingest(DOMAIN_CORPUS)
    
    stats = model.ingester.get_statistics()
    print(f"Learned {stats['total_concepts']} concepts")
    print(f"Frames: {stats['total_frames']}")
    print(f"Cross-domain concepts: {stats['cross_domain_concepts']}")
    print()
    print("Sentences by domain:")
    for domain, count in stats['domains'].items():
        print(f"  {domain}: {count}")
    print()
    
    # Domain-specific generation
    print("=" * 70)
    print("DOMAIN-SPECIFIC GENERATION")
    print("=" * 70)
    print()
    
    for domain in ['sherlock', 'alice', 'pride', 'hamlet']:
        print(f"Domain: {domain}")
        model.set_style(style='neutral', certainty='neutral')
        model.set_morphology(tense='present')
        text = model.generate(domain=domain, num_sentences=2)
        print(f"  {text}")
        print()
    
    # Domain-specific Q&A
    print("=" * 70)
    print("DOMAIN-SPECIFIC Q&A")
    print("=" * 70)
    print()
    
    questions = [
        "Who is Holmes?",
        "Who is Alice?",
        "Who is Hamlet?",
        "What does Holmes do?",
        "What does Darcy do?",
    ]
    
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {model.ask(q)}")
        print()
    
    # Cross-domain queries
    print("=" * 70)
    print("CROSS-DOMAIN QUERIES")
    print("=" * 70)
    print()
    
    cross_questions = [
        "Who examines?",
        "Who watches?",
        "Who kills?",
        "Who loves?",
    ]
    
    for q in cross_questions:
        print(f"Q: {q}")
        print(f"A: {model.ask(q)}")
        print()
    
    # Domain transfer
    print("=" * 70)
    print("DOMAIN TRANSFER")
    print("=" * 70)
    print()
    
    transfer_questions = [
        "What would Alice do in a mystery?",
        "What would Holmes do in a romance?",
        "What would Darcy do in a tragedy?",
    ]
    
    for q in transfer_questions:
        print(f"Q: {q}")
        print(f"A: {model.ask(q)}")
        print()
    
    # Entity comparison
    print("=" * 70)
    print("CROSS-DOMAIN COMPARISON")
    print("=" * 70)
    print()
    
    print("Q: Compare Holmes and Hamlet")
    print(f"A: {model._compare_entities('holmes', 'hamlet')}")
    print()
    
    print("Q: Compare Watson and Nick")
    print(f"A: {model._compare_entities('watson', 'nick')}")
    print()
    
    return model


if __name__ == "__main__":
    model = run_demo()
