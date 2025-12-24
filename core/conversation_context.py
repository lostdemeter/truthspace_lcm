"""
Conversation Context - Tracks entities, topics, and state across turns.

Provides context awareness for:
1. Entity tracking - Remember what entities have been discussed
2. Topic continuity - Track the current topic of conversation
3. Reference resolution - Help resolve "it", "that", "the previous" etc.
4. Conversation state - Track what the user has asked about
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from collections import deque
from datetime import datetime


@dataclass
class EntityMention:
    """A mention of an entity in conversation."""
    name: str
    normalized: str  # Lowercase, canonical form
    turn: int
    timestamp: datetime = field(default_factory=datetime.now)
    context: Optional[str] = None  # The sentence/query it appeared in
    
    def __hash__(self):
        return hash(self.normalized)


@dataclass
class TopicState:
    """Current topic of conversation."""
    topic: str
    domain: Optional[str] = None  # e.g., "Sherlock Holmes", "Pride and Prejudice"
    entities: list[str] = field(default_factory=list)
    turn_started: int = 0
    
    def is_active(self, current_turn: int, max_turns: int = 5) -> bool:
        """Check if topic is still active (within last N turns)."""
        return current_turn - self.turn_started <= max_turns


class ConversationContext:
    """
    Manages conversation context across turns.
    
    Tracks:
    - Recently mentioned entities
    - Current topic/domain
    - Conversation history summary
    """
    
    def __init__(self, max_entities: int = 20, max_history: int = 10):
        self.turn = 0
        self.entities: deque[EntityMention] = deque(maxlen=max_entities)
        self.current_topic: Optional[TopicState] = None
        self.history: deque[dict] = deque(maxlen=max_history)
        
        # Entity normalization map
        self.entity_aliases = {
            'sherlock': 'holmes',
            'sherlock holmes': 'holmes',
            'dr watson': 'watson',
            'doctor watson': 'watson',
            'john watson': 'watson',
            'professor moriarty': 'moriarty',
            'james moriarty': 'moriarty',
            'irene adler': 'irene',
            'the woman': 'irene',
            'mycroft holmes': 'mycroft',
            'inspector lestrade': 'lestrade',
            'mr darcy': 'darcy',
            'fitzwilliam darcy': 'darcy',
            'elizabeth bennet': 'elizabeth',
            'lizzy': 'elizabeth',
            'miss bennet': 'elizabeth',
            'jane bennet': 'jane',
            'mr bingley': 'bingley',
            'charles bingley': 'bingley',
            'mr wickham': 'wickham',
            'george wickham': 'wickham',
            'lydia bennet': 'lydia',
        }
        
        # Domain detection
        self.domain_entities = {
            'holmes': 'Sherlock Holmes',
            'watson': 'Sherlock Holmes',
            'moriarty': 'Sherlock Holmes',
            'irene': 'Sherlock Holmes',
            'mycroft': 'Sherlock Holmes',
            'lestrade': 'Sherlock Holmes',
            'darcy': 'Pride and Prejudice',
            'elizabeth': 'Pride and Prejudice',
            'jane': 'Pride and Prejudice',
            'bingley': 'Pride and Prejudice',
            'wickham': 'Pride and Prejudice',
            'lydia': 'Pride and Prejudice',
        }
    
    def add_turn(self, user_message: str, assistant_response: str):
        """Record a conversation turn."""
        self.turn += 1
        
        # Extract and track entities from user message
        entities = self._extract_entities(user_message)
        for entity in entities:
            self.entities.append(EntityMention(
                name=entity,
                normalized=self._normalize_entity(entity),
                turn=self.turn,
                context=user_message
            ))
        
        # Update topic if needed
        self._update_topic(entities)
        
        # Store in history
        self.history.append({
            'turn': self.turn,
            'user': user_message,
            'assistant': assistant_response,
            'entities': entities,
            'timestamp': datetime.now()
        })
    
    def get_recent_entities(self, n: int = 5) -> list[str]:
        """Get the N most recently mentioned entities."""
        seen = set()
        result = []
        for mention in reversed(self.entities):
            if mention.normalized not in seen:
                seen.add(mention.normalized)
                result.append(mention.normalized)
                if len(result) >= n:
                    break
        return result
    
    def get_last_entity(self, gender: Optional[str] = None) -> Optional[str]:
        """
        Get the most recently mentioned entity.
        
        Args:
            gender: Optional filter ('male', 'female')
        """
        gender_map = {
            'holmes': 'male', 'watson': 'male', 'moriarty': 'male',
            'mycroft': 'male', 'lestrade': 'male',
            'irene': 'female',
            'darcy': 'male', 'bingley': 'male', 'wickham': 'male',
            'elizabeth': 'female', 'jane': 'female', 'lydia': 'female',
        }
        
        for mention in reversed(self.entities):
            if gender is None:
                return mention.normalized
            if gender_map.get(mention.normalized) == gender:
                return mention.normalized
        
        return None
    
    def get_current_domain(self) -> Optional[str]:
        """Get the current conversation domain."""
        if self.current_topic:
            return self.current_topic.domain
        return None
    
    def resolve_reference(self, reference: str) -> Optional[str]:
        """
        Resolve a reference like "he", "she", "it", "that character".
        
        Returns the normalized entity name or None.
        """
        ref_lower = reference.lower().strip()
        
        # Pronoun resolution
        if ref_lower in ['he', 'him', 'his']:
            return self.get_last_entity(gender='male')
        elif ref_lower in ['she', 'her', 'hers']:
            return self.get_last_entity(gender='female')
        elif ref_lower in ['it', 'that', 'this']:
            return self.get_last_entity()
        elif ref_lower in ['they', 'them', 'their']:
            # Return most recent entities
            recent = self.get_recent_entities(2)
            return recent[0] if recent else None
        
        # "that character", "the detective", etc.
        if 'character' in ref_lower or 'person' in ref_lower:
            return self.get_last_entity()
        if 'detective' in ref_lower:
            return 'holmes'
        if 'doctor' in ref_lower:
            return 'watson'
        
        return None
    
    def _extract_entities(self, text: str) -> list[str]:
        """Extract entity mentions from text."""
        text_lower = text.lower()
        entities = []
        
        # Check for known entities
        for alias, canonical in self.entity_aliases.items():
            if alias in text_lower:
                if canonical not in entities:
                    entities.append(canonical)
        
        # Also check canonical names directly
        for canonical in self.domain_entities.keys():
            if canonical in text_lower and canonical not in entities:
                entities.append(canonical)
        
        return entities
    
    def _normalize_entity(self, entity: str) -> str:
        """Normalize an entity name to canonical form."""
        entity_lower = entity.lower()
        return self.entity_aliases.get(entity_lower, entity_lower)
    
    def _update_topic(self, entities: list[str]):
        """Update the current topic based on mentioned entities."""
        if not entities:
            return
        
        # Determine domain from entities
        domains = [self.domain_entities.get(e) for e in entities]
        domains = [d for d in domains if d]
        
        if domains:
            # Use the most common domain
            domain = max(set(domains), key=domains.count)
            
            if self.current_topic is None or self.current_topic.domain != domain:
                self.current_topic = TopicState(
                    topic=entities[0],
                    domain=domain,
                    entities=entities,
                    turn_started=self.turn
                )
            else:
                # Same domain, update entities
                self.current_topic.entities.extend(entities)
    
    def get_context_summary(self) -> dict:
        """Get a summary of the current conversation context."""
        return {
            'turn': self.turn,
            'recent_entities': self.get_recent_entities(),
            'current_domain': self.get_current_domain(),
            'topic': self.current_topic.topic if self.current_topic else None,
            'history_length': len(self.history),
        }
    
    def clear(self):
        """Clear all context."""
        self.turn = 0
        self.entities.clear()
        self.current_topic = None
        self.history.clear()


# Global instance
_context: Optional[ConversationContext] = None

def get_conversation_context() -> ConversationContext:
    """Get the global conversation context."""
    global _context
    if _context is None:
        _context = ConversationContext()
    return _context
