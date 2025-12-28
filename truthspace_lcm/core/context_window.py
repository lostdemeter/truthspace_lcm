#!/usr/bin/env python3
"""
Context Window Extraction for GeometricLCM

Instead of extracting from single sentences, this module uses sliding windows
to capture cross-sentence context and co-occurrence relationships.

Key Concepts:
1. Paragraph Windows - Group sentences into overlapping windows
2. Co-occurrence Tracking - Track which words/entities appear together
3. Distance-Weighted Attention - Closer words have stronger association
4. Entity-Centric Aggregation - Build entity profiles from all windows

This addresses the limitation that definitional information often spans
multiple sentences:
  "He was a consulting detective. Holmes had made this his profession."
  
Single-sentence extraction misses that Holmes = consulting detective.
Window-based extraction captures this co-occurrence.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Counter as CounterType
from collections import Counter, defaultdict
import math


# Words that indicate roles/professions (explicit labels - rare in literature)
ROLE_WORDS = {
    'detective', 'doctor', 'physician', 'captain', 'colonel', 'professor',
    'inspector', 'gentleman', 'lady', 'servant', 'butler', 'maid',
    'king', 'queen', 'prince', 'princess', 'lord', 'duke', 'earl',
    'lawyer', 'banker', 'merchant', 'sailor', 'soldier', 'officer',
    'villain', 'criminal', 'thief', 'murderer', 'victim',
    'friend', 'companion', 'partner', 'enemy', 'rival',
    'husband', 'wife', 'father', 'mother', 'son', 'daughter',
    'brother', 'sister', 'uncle', 'aunt', 'cousin',
    'narrator', 'protagonist', 'hero', 'heroine',
}

# Action words that IMPLY roles (common in literature)
# Literature shows characters doing things rather than labeling them
ACTION_TO_ROLE = {
    # Detective-like actions
    'examined': 'investigator', 'investigated': 'investigator', 
    'deduced': 'investigator', 'observed': 'investigator',
    'inspected': 'investigator', 'searched': 'investigator',
    'case': 'investigator',  # "the case" implies investigation
    
    # Medical actions
    'treated': 'medical', 'diagnosed': 'medical', 'prescribed': 'medical',
    
    # Leadership actions
    'commanded': 'leader', 'ordered': 'leader', 'led': 'leader',
    
    # Speaking/social actions (common for main characters)
    'remarked': 'speaker', 'exclaimed': 'speaker', 'cried': 'speaker',
    'answered': 'speaker', 'replied': 'speaker',
}

# Words that indicate the character is a MAIN character (high agency)
PROTAGONIST_INDICATORS = {
    'remarked', 'observed', 'answered', 'replied', 'cried', 'exclaimed',
    'laughed', 'smiled', 'nodded', 'shook', 'turned', 'rose', 'sat',
    'walked', 'ran', 'came', 'went', 'took', 'gave', 'held', 'looked',
}

# Words that indicate qualities/traits
QUALITY_WORDS = {
    'brilliant', 'clever', 'intelligent', 'wise', 'cunning', 'shrewd',
    'observant', 'perceptive', 'analytical', 'logical', 'rational',
    'kind', 'gentle', 'compassionate', 'loving', 'caring', 'warm',
    'cold', 'cruel', 'harsh', 'stern', 'strict', 'severe',
    'proud', 'humble', 'arrogant', 'modest', 'shy', 'bold', 'brave',
    'loyal', 'faithful', 'treacherous', 'deceitful',
    'strong', 'weak', 'handsome', 'beautiful', 'ugly',
    'witty', 'charming', 'mysterious', 'eccentric', 'peculiar',
    'tall', 'short', 'thin', 'old', 'young',
    'rich', 'poor', 'wealthy', 'noble',
}

# Common words to skip in co-occurrence
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else',
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'this', 'that', 'these', 'those', 'it', 'its',
    'he', 'she', 'him', 'her', 'his', 'hers', 'they', 'them', 'their',
    'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your',
    'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from',
    'as', 'so', 'no', 'not', 'yes', 'very', 'much', 'more',
    'said', 'says', 'say', 'asked', 'replied', 'answered',
    'one', 'two', 'first', 'last', 'all', 'some', 'any',
}


@dataclass
class EntityContext:
    """Aggregated context for an entity from all windows."""
    entity: str
    mention_count: int = 0
    
    # Co-occurring words with weighted counts
    role_cooccurrence: Counter = field(default_factory=Counter)
    quality_cooccurrence: Counter = field(default_factory=Counter)
    entity_cooccurrence: Counter = field(default_factory=Counter)
    action_cooccurrence: Counter = field(default_factory=Counter)  # Actions near entity
    
    # Sample contexts
    sample_windows: List[str] = field(default_factory=list)
    
    # Inferred attributes
    primary_role: Optional[str] = None
    qualities: List[str] = field(default_factory=list)
    related_entities: List[Tuple[str, float]] = field(default_factory=list)
    is_protagonist: bool = False
    
    def infer_attributes(self):
        """Infer role, qualities, and relationships from co-occurrence."""
        # Infer role from actions first (literature shows, doesn't tell)
        # Action-based inference is more reliable for literary texts
        action_role = None
        action_role_score = 0
        if self.action_cooccurrence:
            role_votes = Counter()
            for action, count in self.action_cooccurrence.items():
                if action in ACTION_TO_ROLE:
                    role_votes[ACTION_TO_ROLE[action]] += count
            if role_votes:
                action_role, action_role_score = role_votes.most_common(1)[0]
        
        # Check explicit role words (rare in literature but definitive)
        explicit_role = None
        explicit_role_score = 0
        if self.role_cooccurrence:
            # Filter out generic relationship words
            generic_roles = {'friend', 'companion', 'partner', 'enemy', 'rival'}
            specific_roles = {r: c for r, c in self.role_cooccurrence.items() 
                            if r not in generic_roles}
            if specific_roles:
                explicit_role, explicit_role_score = Counter(specific_roles).most_common(1)[0]
        
        # Choose role: prefer specific explicit roles, then action-inferred, then generic
        if explicit_role and explicit_role_score > 5:
            self.primary_role = explicit_role
        elif action_role and action_role_score > 10:
            self.primary_role = action_role
        elif self.role_cooccurrence:
            self.primary_role = self.role_cooccurrence.most_common(1)[0][0]
        else:
            self.primary_role = "character"
        
        # Check if protagonist (high agency character)
        protagonist_score = sum(
            self.action_cooccurrence.get(action, 0) 
            for action in PROTAGONIST_INDICATORS
        )
        self.is_protagonist = protagonist_score > self.mention_count * 0.1
        
        # Top qualities
        self.qualities = [q for q, _ in self.quality_cooccurrence.most_common(3)]
        
        # Related entities (filter stop words, normalize by mention count)
        if self.entity_cooccurrence:
            # Filter out common words that leaked through
            filtered = {
                e: c for e, c in self.entity_cooccurrence.items()
                if e not in STOP_WORDS and len(e) > 2
            }
            total = sum(filtered.values()) if filtered else 1
            self.related_entities = [
                (e, count / total) 
                for e, count in Counter(filtered).most_common(5)
            ]


class ContextWindowExtractor:
    """
    Extract entity context using sliding windows over text.
    
    Instead of single-sentence extraction, this captures co-occurrence
    patterns across sentence boundaries.
    """
    
    def __init__(self, 
                 window_size: int = 5,
                 decay_factor: float = 0.7):
        """
        Args:
            window_size: Number of sentences per window
            decay_factor: Weight decay for distance (0.7 = 30% decay per sentence)
        """
        self.window_size = window_size
        self.decay_factor = decay_factor
        
        # Known entities (capitalized words that appear frequently)
        self.known_entities: Set[str] = set()
        
        # Entity contexts
        self.entity_contexts: Dict[str, EntityContext] = {}
    
    def extract_from_text(self, text: str, source: str = "") -> Dict[str, EntityContext]:
        """
        Extract entity contexts from text using sliding windows.
        
        Args:
            text: Full text to process
            source: Source name for reference
            
        Returns:
            Dict mapping entity names to their contexts
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        # First pass: identify entities (frequent capitalized words)
        self._identify_entities(sentences)
        
        # Second pass: extract co-occurrence in windows
        self._extract_windows(sentences)
        
        # Third pass: infer attributes
        for ctx in self.entity_contexts.values():
            ctx.infer_attributes()
        
        return self.entity_contexts
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def _identify_entities(self, sentences: List[str], min_count: int = 3):
        """Identify entities as frequently occurring capitalized words."""
        word_counts = Counter()
        
        for sent in sentences:
            words = re.findall(r'\b[A-Z][a-z]+\b', sent)
            for word in words:
                word_lower = word.lower()
                if word_lower not in STOP_WORDS and len(word) > 2:
                    word_counts[word_lower] += 1
        
        # Entities are words that appear at least min_count times
        self.known_entities = {
            word for word, count in word_counts.items() 
            if count >= min_count
        }
    
    def _extract_windows(self, sentences: List[str]):
        """Extract co-occurrence from sliding windows."""
        for i in range(len(sentences)):
            # Get window of sentences
            window_start = max(0, i - self.window_size // 2)
            window_end = min(len(sentences), i + self.window_size // 2 + 1)
            window = sentences[window_start:window_end]
            
            # Get all words in window with position-based weights
            window_text = ' '.join(window)
            center_sentence = sentences[i]
            
            # Find entities in center sentence
            center_entities = self._find_entities_in_text(center_sentence)
            
            if not center_entities:
                continue
            
            # Find co-occurring words in full window
            window_words = self._tokenize(window_text)
            
            for entity in center_entities:
                # Get or create context
                if entity not in self.entity_contexts:
                    self.entity_contexts[entity] = EntityContext(entity=entity)
                ctx = self.entity_contexts[entity]
                
                ctx.mention_count += 1
                
                # Track co-occurrence with distance weighting
                for j, word in enumerate(window_words):
                    word_lower = word.lower()
                    
                    # Skip the entity itself and stop words
                    if word_lower == entity or word_lower in STOP_WORDS:
                        continue
                    
                    # Calculate distance-based weight
                    # (simplified: use 1.0 for now, can add position tracking later)
                    weight = 1.0
                    
                    # Categorize the word
                    if word_lower in ROLE_WORDS:
                        ctx.role_cooccurrence[word_lower] += weight
                    elif word_lower in QUALITY_WORDS:
                        ctx.quality_cooccurrence[word_lower] += weight
                    elif word_lower in ACTION_TO_ROLE or word_lower in PROTAGONIST_INDICATORS:
                        ctx.action_cooccurrence[word_lower] += weight
                    elif word_lower in self.known_entities and word_lower != entity:
                        ctx.entity_cooccurrence[word_lower] += weight
                
                # Store sample window
                if len(ctx.sample_windows) < 3:
                    ctx.sample_windows.append(center_sentence[:200])
    
    def _find_entities_in_text(self, text: str) -> List[str]:
        """Find known entities in text."""
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        return [w.lower() for w in words if w.lower() in self.known_entities]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def get_entity_summary(self, entity: str) -> Optional[str]:
        """Generate a natural language summary for an entity."""
        entity_lower = entity.lower()
        if entity_lower not in self.entity_contexts:
            return None
        
        ctx = self.entity_contexts[entity_lower]
        
        parts = []
        
        # Opening with role
        name = entity.title()
        if ctx.primary_role:
            parts.append(f"{name} is a {ctx.primary_role}.")
        else:
            parts.append(f"{name} is a character in the text.")
        
        # Qualities
        if ctx.qualities:
            qual_str = ', '.join(ctx.qualities[:2])
            if len(ctx.qualities) > 2:
                qual_str += f", and {ctx.qualities[2]}"
            parts.append(f"They are described as {qual_str}.")
        
        # Relationships
        if ctx.related_entities:
            top_related = ctx.related_entities[0][0]
            parts.append(f"They are closely associated with {top_related.title()}.")
        
        return ' '.join(parts)


def extract_entity_contexts(text: str, source: str = "") -> Dict[str, EntityContext]:
    """
    Convenience function to extract entity contexts from text.
    
    Args:
        text: Full text to process
        source: Source name
        
    Returns:
        Dict mapping entity names to their contexts
    """
    extractor = ContextWindowExtractor()
    return extractor.extract_from_text(text, source)


def build_cooccurrence_matrix(text: str, 
                               window_size: int = 5) -> Dict[str, Counter]:
    """
    Build a co-occurrence matrix from text.
    
    Returns dict mapping each entity to a Counter of co-occurring words.
    """
    extractor = ContextWindowExtractor(window_size=window_size)
    contexts = extractor.extract_from_text(text)
    
    # Combine all co-occurrence types
    matrix = {}
    for entity, ctx in contexts.items():
        combined = Counter()
        combined.update(ctx.role_cooccurrence)
        combined.update(ctx.quality_cooccurrence)
        combined.update(ctx.entity_cooccurrence)
        matrix[entity] = combined
    
    return matrix
