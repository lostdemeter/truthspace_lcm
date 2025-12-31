"""
Text Processing Gears

Gears for text analysis that can be composed into processing pipelines:
- StopwordGear: Emergent stopword detection using frequency analysis
- GenderGear: Gender detection using SemanticQuaternion x-axis
- PronounResolutionGear: Resolves pronouns to their antecedents
- ThoughtChainingGear: Composes related facts into coherent explanations

Author: Lesley Gushurst
License: GPLv3
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

from truthspace_lcm.gears.core import Gear, GearState


@dataclass
class EntityMention:
    """A mention of an entity in text."""
    text: str
    position: int
    gender: Optional[str] = None  # 'male', 'female', 'neutral', None
    is_pronoun: bool = False
    resolved_to: Optional[str] = None


class StopwordGear(Gear):
    """
    Emergent stopword detection using Zipf's law.
    
    Instead of hardcoded lists, detects stopwords by:
    1. High frequency (appears in many contexts)
    2. Low information content (short, common patterns)
    3. Uniform distribution across topics
    
    The key insight: stopwords are structurally identifiable by their
    frequency distribution, not by memorizing a list.
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("StopwordGear", ratio)
        
        # Learned word frequencies
        self.word_counts: Counter = Counter()
        self.total_words: int = 0
        self.document_counts: Counter = Counter()  # How many docs each word appears in
        self.total_documents: int = 0
        
        # Thresholds (can be tuned)
        self.frequency_threshold: float = 0.01  # Top 1% by frequency
        self.length_threshold: int = 4  # Short words more likely stopwords
        self.document_frequency_threshold: float = 0.5  # Appears in 50%+ of docs
        
        # Cache of detected stopwords
        self._stopwords: Optional[Set[str]] = None
    
    def learn_from_text(self, text: str, document_id: str = None):
        """Learn word frequencies from text."""
        words = re.findall(r'\b[a-z]+\b', text.lower())
        self.word_counts.update(words)
        self.total_words += len(words)
        
        if document_id:
            unique_words = set(words)
            self.document_counts.update(unique_words)
            self.total_documents += 1
        
        # Invalidate cache
        self._stopwords = None
    
    def get_stopwords(self) -> Set[str]:
        """Get the set of detected stopwords."""
        if self._stopwords is not None:
            return self._stopwords
        
        if self.total_words == 0:
            # Fallback to common stopwords if no learning has occurred
            return self._default_stopwords()
        
        stopwords = set()
        
        for word, count in self.word_counts.items():
            frequency = count / self.total_words
            
            # High frequency words
            if frequency > self.frequency_threshold:
                stopwords.add(word)
                continue
            
            # Short, high-frequency words
            if len(word) <= self.length_threshold and frequency > self.frequency_threshold / 2:
                stopwords.add(word)
                continue
            
            # Words appearing in many documents (if document tracking enabled)
            if self.total_documents > 0:
                doc_freq = self.document_counts.get(word, 0) / self.total_documents
                if doc_freq > self.document_frequency_threshold and len(word) <= 5:
                    stopwords.add(word)
        
        self._stopwords = stopwords
        return stopwords
    
    def _default_stopwords(self) -> Set[str]:
        """Fallback stopwords when no learning has occurred."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'our', 'their', 'what', 'which', 'who', 'whom', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        }
    
    def is_stopword(self, word: str) -> bool:
        """Check if a word is a stopword."""
        return word.lower() in self.get_stopwords()
    
    def forward(self, state: GearState) -> GearState:
        """Mark stopwords in the state."""
        text = state.metadata.get('text', '')
        if text:
            words = text.lower().split()
            stopwords = self.get_stopwords()
            content_words = [w for w in words if w not in stopwords]
            state.metadata['content_words'] = content_words
            state.metadata['stopword_ratio'] = 1 - len(content_words) / max(len(words), 1)
        return state


class GenderGear(Gear):
    """
    Gender detection using semantic patterns.
    
    Uses:
    1. Known gender pairs (king/queen, actor/actress)
    2. Suffix patterns (-ess, -ine for female)
    3. SemanticQuaternion x-axis if available
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("GenderGear", ratio)
        
        # Known gender mappings
        self.male_words: Set[str] = {
            'he', 'him', 'his', 'himself', 'man', 'men', 'boy', 'boys',
            'father', 'son', 'brother', 'uncle', 'nephew', 'husband',
            'king', 'prince', 'lord', 'sir', 'mr', 'gentleman',
            'actor', 'waiter', 'host', 'hero', 'god',
        }
        
        self.female_words: Set[str] = {
            'she', 'her', 'hers', 'herself', 'woman', 'women', 'girl', 'girls',
            'mother', 'daughter', 'sister', 'aunt', 'niece', 'wife',
            'queen', 'princess', 'lady', 'madam', 'mrs', 'miss', 'ms',
            'actress', 'waitress', 'hostess', 'heroine', 'goddess',
        }
        
        # Female suffixes
        self.female_suffixes = ['ess', 'ress', 'ine', 'ette', 'trix']
        
        # Learned gender associations
        self.learned_genders: Dict[str, str] = {}
    
    def detect_gender(self, word: str) -> Optional[str]:
        """Detect gender of a word. Returns 'male', 'female', or None."""
        word_lower = word.lower()
        
        # Check known words
        if word_lower in self.male_words:
            return 'male'
        if word_lower in self.female_words:
            return 'female'
        
        # Check learned associations
        if word_lower in self.learned_genders:
            return self.learned_genders[word_lower]
        
        # Check suffixes
        for suffix in self.female_suffixes:
            if word_lower.endswith(suffix):
                return 'female'
        
        return None
    
    def learn_gender(self, word: str, gender: str):
        """Learn a gender association for a word."""
        self.learned_genders[word.lower()] = gender
    
    def forward(self, state: GearState) -> GearState:
        """Add gender information to state."""
        entity = state.entity
        if entity:
            gender = self.detect_gender(entity)
            state.metadata['gender'] = gender
        return state


class PronounResolutionGear(Gear):
    """
    Resolves pronouns to their antecedents.
    
    Tracks the most recent named entities and resolves pronouns
    based on gender and recency.
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("PronounResolutionGear", ratio)
        
        self.gender_gear = GenderGear()
        
        # Pronoun categories
        self.male_pronouns = {'he', 'him', 'his', 'himself'}
        self.female_pronouns = {'she', 'her', 'hers', 'herself'}
        self.neutral_pronouns = {'it', 'its', 'itself', 'they', 'them', 'their', 'themselves'}
        
        # Current antecedent tracking
        self.last_male: Optional[str] = None
        self.last_female: Optional[str] = None
        self.last_entity: Optional[str] = None
        
        # Entity mention history
        self.mentions: List[EntityMention] = []
    
    def reset(self):
        """Reset antecedent tracking (e.g., at paragraph boundaries)."""
        self.last_male = None
        self.last_female = None
        self.last_entity = None
        self.mentions = []
    
    def process_entity(self, entity: str, position: int = 0) -> EntityMention:
        """Process an entity mention, resolving pronouns if needed."""
        entity_lower = entity.lower()
        
        # Check if it's a pronoun
        is_pronoun = entity_lower in (self.male_pronouns | self.female_pronouns | self.neutral_pronouns)
        
        if is_pronoun:
            # Resolve pronoun
            resolved = None
            if entity_lower in self.male_pronouns and self.last_male:
                resolved = self.last_male
            elif entity_lower in self.female_pronouns and self.last_female:
                resolved = self.last_female
            elif entity_lower in self.neutral_pronouns and self.last_entity:
                resolved = self.last_entity
            
            mention = EntityMention(
                text=entity,
                position=position,
                gender=self._pronoun_gender(entity_lower),
                is_pronoun=True,
                resolved_to=resolved,
            )
        else:
            # Named entity - update tracking
            gender = self.gender_gear.detect_gender(entity)
            
            self.last_entity = entity_lower
            if gender == 'male':
                self.last_male = entity_lower
            elif gender == 'female':
                self.last_female = entity_lower
            
            mention = EntityMention(
                text=entity,
                position=position,
                gender=gender,
                is_pronoun=False,
                resolved_to=None,
            )
        
        self.mentions.append(mention)
        return mention
    
    def _pronoun_gender(self, pronoun: str) -> Optional[str]:
        """Get the gender of a pronoun."""
        if pronoun in self.male_pronouns:
            return 'male'
        elif pronoun in self.female_pronouns:
            return 'female'
        return 'neutral'
    
    def resolve(self, pronoun: str) -> Optional[str]:
        """Resolve a pronoun to its antecedent."""
        pronoun_lower = pronoun.lower()
        
        if pronoun_lower in self.male_pronouns:
            return self.last_male
        elif pronoun_lower in self.female_pronouns:
            return self.last_female
        elif pronoun_lower in self.neutral_pronouns:
            return self.last_entity
        
        return None
    
    def forward(self, state: GearState) -> GearState:
        """Resolve pronouns in the state."""
        entity = state.entity
        if entity:
            mention = self.process_entity(entity)
            if mention.is_pronoun and mention.resolved_to:
                state.metadata['original_entity'] = entity
                state.metadata['resolved_entity'] = mention.resolved_to
                state.entity = mention.resolved_to
        return state


class ThoughtChainingGear(Gear):
    """
    Composes related facts into coherent explanations.
    
    Given a set of facts about an entity, chains them together
    into a narrative explanation.
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("ThoughtChainingGear", ratio)
        
        # Connectors for different relationship types
        self.connectors = {
            'action': ['who', 'that', 'which'],
            'attribute': ['is', 'was', 'appears to be'],
            'relation': ['associated with', 'connected to', 'related to'],
            'sequence': ['then', 'afterwards', 'subsequently'],
            'contrast': ['however', 'but', 'although'],
            'cause': ['because', 'since', 'as a result of'],
        }
    
    def chain_facts(self, entity: str, facts: List[Dict[str, Any]]) -> str:
        """
        Chain facts about an entity into a coherent explanation.
        
        Args:
            entity: The main entity
            facts: List of fact dicts with 'type', 'content', 'related_entities'
        
        Returns:
            A composed explanation string
        """
        if not facts:
            return f"I don't have detailed information about {entity}."
        
        parts = []
        
        # Start with the entity
        parts.append(f"{entity.title()}")
        
        # Group facts by type
        actions = [f for f in facts if f.get('type') == 'action']
        attributes = [f for f in facts if f.get('type') == 'attribute']
        relations = [f for f in facts if f.get('type') == 'relation']
        
        # Add actions
        if actions:
            action_words = [a.get('content', '') for a in actions[:3]]
            action_str = ', '.join(action_words)
            parts.append(f"is known to {action_str}")
        
        # Add relations
        if relations:
            related = [r.get('related_entity', '') for r in relations[:3]]
            related_str = ', '.join([r.title() for r in related if r])
            if related_str:
                parts.append(f"and is associated with {related_str}")
        
        # Add attributes
        if attributes:
            attr_words = [a.get('content', '') for a in attributes[:2]]
            if attr_words:
                parts.append(f"characterized by {', '.join(attr_words)}")
        
        return ' '.join(parts) + '.'
    
    def compose_from_profile(self, entity: str, profile: Dict[str, Any]) -> str:
        """
        Compose an explanation from an entity profile.
        
        Args:
            entity: The entity name
            profile: Dict with 'actions', 'targets', 'related_entities'
        
        Returns:
            A composed explanation
        """
        parts = [f"{entity.title()}"]
        
        actions = profile.get('actions', {})
        targets = profile.get('targets', {})
        related = profile.get('related_entities', {})
        
        if actions:
            # Convert past tense verbs to base form for "is known to X"
            action_list = [self._to_base_verb(a) for a in list(actions.keys())[:3]]
            parts.append(f"is known to {', '.join(action_list)}")
        
        if related:
            related_list = [r.title() for r in list(related.keys())[:3]]
            parts.append(f"and is associated with {', '.join(related_list)}")
        
        if not actions and not related:
            return f"I have limited information about {entity}."
        
        return ' '.join(parts) + '.'
    
    def _to_base_verb(self, verb: str) -> str:
        """Convert a verb to its base form (simple heuristic)."""
        verb = verb.lower()
        
        # Common irregular past tense -> base
        irregulars = {
            'sat': 'sit', 'found': 'find', 'said': 'say', 'saw': 'see',
            'went': 'go', 'came': 'come', 'made': 'make', 'took': 'take',
            'got': 'get', 'gave': 'give', 'thought': 'think', 'told': 'tell',
            'felt': 'feel', 'knew': 'know', 'left': 'leave', 'heard': 'hear',
            'began': 'begin', 'kept': 'keep', 'held': 'hold', 'stood': 'stand',
            'ran': 'run', 'fell': 'fall', 'rose': 'rise', 'lay': 'lie',
            'slept': 'sleep', 'woke': 'wake', 'ate': 'eat', 'drank': 'drink',
            'spoke': 'speak', 'wrote': 'write', 'read': 'read', 'brought': 'bring',
            'caught': 'catch', 'threw': 'throw', 'struck': 'strike', 'hit': 'hit',
        }
        
        if verb in irregulars:
            return irregulars[verb]
        
        # Regular past tense: remove -ed
        if verb.endswith('ed'):
            if verb.endswith('ied'):
                return verb[:-3] + 'y'  # studied -> study
            elif verb.endswith('eed'):
                return verb[:-2]  # agreed -> agree
            elif len(verb) > 4 and verb[-3] == verb[-4]:
                return verb[:-3]  # stopped -> stop
            elif verb.endswith('ged') or verb.endswith('ced') or verb.endswith('sed'):
                return verb[:-1]  # budged -> budge, danced -> dance
            else:
                return verb[:-2]  # walked -> walk
        
        # Present participle: remove -ing
        if verb.endswith('ing'):
            base = verb[:-3]
            if len(base) > 1 and base[-1] == base[-2]:
                return base[:-1]  # running -> run
            return base + 'e' if base.endswith(('tak', 'mak', 'giv')) else base
        
        return verb
    
    def forward(self, state: GearState) -> GearState:
        """Compose thoughts in the state."""
        facts = state.metadata.get('facts', [])
        entity = state.entity
        
        if facts and entity:
            composed = self.chain_facts(entity, facts)
            state.metadata['composed_explanation'] = composed
        
        return state
