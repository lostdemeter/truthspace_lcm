#!/usr/bin/env python3
"""
Conversation Memory: Multi-Turn Dialogue Support for GeometricLCM

This module implements conversation memory that enables:
1. Multi-turn dialogue with context retention
2. Pronoun resolution (he/she/it → focus entity)
3. Context decay using φ^(-n) weighting
4. Coreference tracking across turns

The key insight: Conversation is a GRAPH of connected concept frames,
not a sequence of independent questions.

Author: Lesley Gushurst
License: GPLv3
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# Golden ratio for decay
PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# PRONOUN MAPPINGS
# =============================================================================

# Pronouns that refer to the focus entity
SUBJECT_PRONOUNS = {'he', 'she', 'it', 'they'}
OBJECT_PRONOUNS = {'him', 'her', 'it', 'them'}
POSSESSIVE_PRONOUNS = {'his', 'her', 'hers', 'its', 'their', 'theirs'}

# Gender hints from names (expandable)
MALE_NAMES = {
    'holmes', 'watson', 'darcy', 'bingley', 'wickham', 'collins', 'bennet',
    'moriarty', 'lestrade', 'mycroft', 'stamford', 'gregson',
}
FEMALE_NAMES = {
    'elizabeth', 'jane', 'lydia', 'charlotte', 'catherine', 'georgiana',
    'irene', 'mary', 'kitty', 'caroline',
}


# =============================================================================
# CONVERSATION TURN
# =============================================================================

@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    query: str                      # The user's question
    answer: str                     # The bot's answer
    entity: Optional[str] = None    # The main entity discussed
    axis: Optional[str] = None      # Question type (WHO/WHAT/WHERE/etc.)
    timestamp: int = 0              # Turn number
    
    def to_dict(self) -> dict:
        return {
            'query': self.query,
            'answer': self.answer,
            'entity': self.entity,
            'axis': self.axis,
            'timestamp': self.timestamp,
        }


# =============================================================================
# CONVERSATION MEMORY
# =============================================================================

class ConversationMemory:
    """
    Memory for multi-turn conversations.
    
    Tracks:
    - Recent turns (with decay)
    - Focus entity (current subject of discussion)
    - Entity gender (for pronoun resolution)
    - Coreference chains (what "he/she/it" refers to)
    """
    
    def __init__(self, max_turns: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_turns: Maximum number of turns to remember
        """
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []
        self.focus_entity: Optional[str] = None
        self.focus_gender: Optional[str] = None  # 'male', 'female', or None
        self.entity_genders: Dict[str, str] = {}
        self.turn_counter = 0
    
    def add_turn(self, query: str, answer: str, 
                 entity: Optional[str] = None, 
                 axis: Optional[str] = None):
        """
        Add a conversation turn to memory.
        
        Args:
            query: The user's question
            answer: The bot's answer
            entity: The main entity discussed (if any)
            axis: The question type (WHO/WHAT/WHERE/etc.)
        """
        self.turn_counter += 1
        
        turn = ConversationTurn(
            query=query,
            answer=answer,
            entity=entity,
            axis=axis,
            timestamp=self.turn_counter,
        )
        
        self.turns.append(turn)
        
        # Update focus entity
        if entity:
            self.focus_entity = entity.lower()
            self._infer_gender(entity)
        
        # Trim old turns
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)
    
    def _infer_gender(self, entity: str):
        """Infer gender from entity name."""
        entity_lower = entity.lower()
        
        if entity_lower in self.entity_genders:
            self.focus_gender = self.entity_genders[entity_lower]
        elif entity_lower in MALE_NAMES:
            self.focus_gender = 'male'
            self.entity_genders[entity_lower] = 'male'
        elif entity_lower in FEMALE_NAMES:
            self.focus_gender = 'female'
            self.entity_genders[entity_lower] = 'female'
        else:
            self.focus_gender = None
    
    def resolve_pronouns(self, text: str) -> str:
        """
        Resolve pronouns in text to the focus entity.
        
        Args:
            text: Text potentially containing pronouns
        
        Returns:
            Text with pronouns replaced by entity name
        """
        if not self.focus_entity:
            return text
        
        # Build replacement map based on gender
        replacements = {}
        
        if self.focus_gender == 'male':
            replacements = {
                r'\bhe\b': self.focus_entity.title(),
                r'\bhim\b': self.focus_entity.title(),
                r'\bhis\b': f"{self.focus_entity.title()}'s",
            }
        elif self.focus_gender == 'female':
            replacements = {
                r'\bshe\b': self.focus_entity.title(),
                r'\bher\b': self.focus_entity.title(),
                r'\bhers\b': f"{self.focus_entity.title()}'s",
            }
        else:
            # Unknown gender - replace all
            replacements = {
                r'\bhe\b': self.focus_entity.title(),
                r'\bshe\b': self.focus_entity.title(),
                r'\bhim\b': self.focus_entity.title(),
                r'\bher\b': self.focus_entity.title(),
                r'\bhis\b': f"{self.focus_entity.title()}'s",
                r'\bhers\b': f"{self.focus_entity.title()}'s",
            }
        
        # Also handle "they/them" for any entity
        replacements[r'\bthey\b'] = self.focus_entity.title()
        replacements[r'\bthem\b'] = self.focus_entity.title()
        replacements[r'\btheir\b'] = f"{self.focus_entity.title()}'s"
        
        # Apply replacements (case-insensitive)
        result = text
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def has_pronoun(self, text: str) -> bool:
        """Check if text contains a pronoun that needs resolution."""
        text_lower = text.lower()
        all_pronouns = SUBJECT_PRONOUNS | OBJECT_PRONOUNS | POSSESSIVE_PRONOUNS
        
        for pronoun in all_pronouns:
            if re.search(rf'\b{pronoun}\b', text_lower):
                return True
        return False
    
    def get_context_weight(self, turn_index: int) -> float:
        """
        Get attention weight for a turn based on recency.
        
        Uses φ^(-n) decay where n is distance from current turn.
        More recent turns get higher weight.
        
        Args:
            turn_index: Index in self.turns (0 = oldest)
        
        Returns:
            Weight in [0, 1]
        """
        if not self.turns:
            return 0.0
        
        # Distance from most recent turn
        distance = len(self.turns) - 1 - turn_index
        
        # φ^(-n) decay
        weight = PHI ** (-distance)
        
        return weight
    
    def get_recent_context(self, k: int = 3) -> List[Tuple[ConversationTurn, float]]:
        """
        Get k most recent turns with their attention weights.
        
        Args:
            k: Number of turns to return
        
        Returns:
            List of (turn, weight) tuples, most recent first
        """
        if not self.turns:
            return []
        
        # Get last k turns
        recent = self.turns[-k:]
        
        # Calculate weights
        result = []
        for i, turn in enumerate(recent):
            weight = self.get_context_weight(len(self.turns) - k + i)
            result.append((turn, weight))
        
        # Return most recent first
        return list(reversed(result))
    
    def get_entities_mentioned(self) -> List[str]:
        """Get all entities mentioned in conversation."""
        entities = set()
        for turn in self.turns:
            if turn.entity:
                entities.add(turn.entity.lower())
        return list(entities)
    
    def get_focus_entity(self) -> Optional[str]:
        """Get the current focus entity."""
        return self.focus_entity
    
    def clear(self):
        """Clear conversation memory."""
        self.turns = []
        self.focus_entity = None
        self.focus_gender = None
        self.turn_counter = 0
    
    def get_summary(self) -> str:
        """Get a summary of the conversation so far."""
        if not self.turns:
            return "No conversation yet."
        
        lines = [f"Conversation ({len(self.turns)} turns):"]
        for turn in self.turns[-5:]:  # Last 5 turns
            entity_str = f" [{turn.entity}]" if turn.entity else ""
            lines.append(f"  Q: {turn.query[:50]}...{entity_str}")
            lines.append(f"  A: {turn.answer[:50]}...")
        
        if self.focus_entity:
            lines.append(f"Focus: {self.focus_entity}")
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self.turns)
    
    def __bool__(self) -> bool:
        return len(self.turns) > 0


# =============================================================================
# TEST
# =============================================================================

def test_conversation_memory():
    """Test conversation memory functionality."""
    print("=== ConversationMemory Test ===\n")
    
    memory = ConversationMemory(max_turns=10)
    
    # Simulate a conversation
    print("## Simulating Conversation")
    print()
    
    # Turn 1
    memory.add_turn(
        query="Who is Holmes?",
        answer="Holmes is a brilliant detective from Sherlock Holmes.",
        entity="holmes",
        axis="WHO"
    )
    print(f"Turn 1: Who is Holmes?")
    print(f"  Focus entity: {memory.focus_entity}")
    print(f"  Focus gender: {memory.focus_gender}")
    print()
    
    # Turn 2 - with pronoun
    query2 = "What did he do?"
    resolved2 = memory.resolve_pronouns(query2)
    print(f"Turn 2: {query2}")
    print(f"  Resolved: {resolved2}")
    print(f"  Has pronoun: {memory.has_pronoun(query2)}")
    
    memory.add_turn(
        query=resolved2,
        answer="Holmes investigated crimes and solved mysteries.",
        entity="holmes",
        axis="WHAT"
    )
    print()
    
    # Turn 3 - switch entity
    memory.add_turn(
        query="Who is Elizabeth?",
        answer="Elizabeth is a witty lady from Pride and Prejudice.",
        entity="elizabeth",
        axis="WHO"
    )
    print(f"Turn 3: Who is Elizabeth?")
    print(f"  Focus entity: {memory.focus_entity}")
    print(f"  Focus gender: {memory.focus_gender}")
    print()
    
    # Turn 4 - pronoun should now refer to Elizabeth
    query4 = "What did she do?"
    resolved4 = memory.resolve_pronouns(query4)
    print(f"Turn 4: {query4}")
    print(f"  Resolved: {resolved4}")
    print()
    
    # Test context weights
    print("## Context Weights (φ^(-n) decay)")
    context = memory.get_recent_context(k=4)
    for turn, weight in context:
        print(f"  {turn.query[:30]}... weight={weight:.3f}")
    print()
    
    # Test summary
    print("## Summary")
    print(memory.get_summary())
    print()
    
    print("=== Test Complete ===")


if __name__ == '__main__':
    test_conversation_memory()
