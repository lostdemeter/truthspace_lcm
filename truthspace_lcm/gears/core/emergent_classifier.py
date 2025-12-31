"""
Emergent Classifier Gear

A meta-gear that discovers word categories from data patterns rather than
hardcoded lists. This generalizes the pattern we keep solving:

Problem → Emergent Signal:
- Stopwords → High frequency, uniform distribution across contexts
- Verbs → Position after subject, morphological patterns (-ed, -ing)
- Gender → Co-occurrence patterns, suffix patterns (-ess, -ine)
- Pronouns → Very high frequency, specific syntactic positions
- Entities → Low frequency, capitalization, specific contexts

The key insight: Word categories are STRUCTURAL, not semantic.
They can be discovered from distributional properties alone.

Protocol for Converting Hardcoded Concepts to Emergent Gears:
1. IDENTIFY the hardcoded concept (stopwords, verbs, etc.)
2. ANALYZE what structural signal distinguishes this category
3. IMPLEMENT a detector based on that signal
4. SEED with a few examples (optional, for bootstrapping)
5. LEARN from data to expand/refine the category
6. VALIDATE against known examples

Author: Lesley Gushurst
License: GPLv3
"""

import re
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable

from truthspace_lcm.gears.core.base import Gear, GearState
from truthspace_lcm.gears.core.gear_message import GearProtocol, GearMessage, MessageIntent


@dataclass
class CategorySignature:
    """
    Signature that defines how to detect a word category.
    
    Each category has multiple signals that can be combined.
    """
    name: str
    
    # Frequency-based signals
    min_frequency: float = 0.0  # Minimum relative frequency
    max_frequency: float = 1.0  # Maximum relative frequency
    
    # Distribution signals
    min_document_frequency: float = 0.0  # Appears in at least X% of documents
    max_document_frequency: float = 1.0
    
    # Position signals
    typical_positions: List[str] = field(default_factory=list)  # 'start', 'middle', 'end'
    position_weight: float = 0.0  # How much position matters
    
    # Morphological signals (suffix/prefix patterns)
    positive_suffixes: List[str] = field(default_factory=list)  # Suffixes that indicate this category
    negative_suffixes: List[str] = field(default_factory=list)  # Suffixes that exclude this category
    suffix_weight: float = 0.0
    
    # Length signals
    min_length: int = 0
    max_length: int = 100
    length_weight: float = 0.0
    
    # Co-occurrence signals
    co_occurs_with: List[str] = field(default_factory=list)  # Words it typically appears with
    cooccurrence_weight: float = 0.0
    
    # Seed examples (for bootstrapping)
    seeds: Set[str] = field(default_factory=set)
    
    def score(self, word: str, stats: 'WordStats') -> float:
        """Score how well a word matches this category signature."""
        score = 0.0
        weights = 0.0
        
        # Seed membership is definitive (strong signal)
        if word.lower() in self.seeds:
            return 1.0  # Definitive match
        
        # Frequency score (only count if we have enough data)
        if stats.total_words > 100 and (self.min_frequency > 0 or self.max_frequency < 1):
            freq = stats.frequency
            if self.min_frequency <= freq <= self.max_frequency:
                score += 1.0
            else:
                score -= 0.5  # Penalty for not matching frequency
            weights += 1.0
        
        # Document frequency score (only if multiple documents)
        if stats.total_documents > 1 and (self.min_document_frequency > 0 or self.max_document_frequency < 1):
            doc_freq = stats.document_frequency
            if self.min_document_frequency <= doc_freq <= self.max_document_frequency:
                score += 1.0
            weights += 1.0
        
        # Position score
        if self.typical_positions and self.position_weight > 0 and stats.position_count > 0:
            pos_match = any(
                (pos == 'start' and stats.avg_position < 0.2) or
                (pos == 'middle' and 0.2 <= stats.avg_position <= 0.8) or
                (pos == 'end' and stats.avg_position > 0.8)
                for pos in self.typical_positions
            )
            if pos_match:
                score += self.position_weight
            weights += self.position_weight
        
        # Suffix score (morphological - always applies)
        if self.positive_suffixes and self.suffix_weight > 0:
            has_positive = any(word.endswith(s) for s in self.positive_suffixes)
            has_negative = any(word.endswith(s) for s in self.negative_suffixes)
            if has_positive and not has_negative:
                score += self.suffix_weight
            elif has_negative:
                score -= self.suffix_weight * 0.5  # Penalty for negative suffix
            weights += self.suffix_weight
        
        # Length score
        if self.length_weight > 0:
            if self.min_length <= len(word) <= self.max_length:
                score += self.length_weight
            else:
                score -= self.length_weight * 0.3  # Small penalty
            weights += self.length_weight
        
        return max(0, score / weights) if weights > 0 else 0.0


@dataclass
class WordStats:
    """Statistics about a word's usage patterns."""
    word: str
    count: int = 0
    document_count: int = 0
    total_words: int = 1
    total_documents: int = 1
    position_sum: float = 0.0
    position_count: int = 0
    
    @property
    def frequency(self) -> float:
        return self.count / max(self.total_words, 1)
    
    @property
    def document_frequency(self) -> float:
        return self.document_count / max(self.total_documents, 1)
    
    @property
    def avg_position(self) -> float:
        return self.position_sum / max(self.position_count, 1)


class EmergentClassifierGear(GearProtocol):
    """
    A gear that discovers word categories from data patterns.
    
    Instead of hardcoding lists like:
        stopwords = {'the', 'a', 'an', ...}
        verbs = {'said', 'went', 'came', ...}
    
    We define signatures that describe the STRUCTURAL properties:
        stopwords: high_frequency + uniform_distribution + short_length
        verbs: position_after_subject + ed/ing_suffix + medium_frequency
    
    The gear then learns which words match each signature.
    """
    
    # Pre-defined signatures for common categories
    STOPWORD_SIGNATURE = CategorySignature(
        name="stopword",
        min_frequency=0.01,  # Top 1% by frequency (stricter)
        min_document_frequency=0.5,  # Appears in 50%+ of contexts
        max_length=4,  # Stopwords are typically very short (the, a, is, etc.)
        length_weight=0.9,  # Very strong length signal - most stopwords are <=4 chars
        seeds={'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
               'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
               'could', 'should', 'may', 'might', 'must', 'can', 'and', 'or',
               'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
               'it', 'its', 'this', 'that', 'i', 'you', 'he', 'she', 'we',
               'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
               'our', 'if', 'so', 'as', 'no', 'not', 'all', 'any', 'each'},
    )
    
    VERB_SIGNATURE = CategorySignature(
        name="verb",
        typical_positions=['middle'],
        position_weight=0.3,
        positive_suffixes=['ed', 'ing'],  # Only strong verb suffixes
        negative_suffixes=['ness', 'ment', 'tion', 'ly', 'le', 'er', 'or', 'al'],
        suffix_weight=0.7,  # Higher weight for morphology
        min_length=3,
        max_length=15,
        length_weight=0.1,
        seeds={'said', 'saw', 'went', 'came', 'made', 'took', 'got', 'gave',
               'found', 'thought', 'told', 'asked', 'looked', 'seemed', 'felt',
               'knew', 'wanted', 'called', 'turned', 'left', 'heard', 'began',
               'stood', 'ran', 'sat', 'walked', 'watched', 'followed', 'stopped',
               'died', 'killed', 'hunted', 'chased', 'caught', 'escaped', 'cried',
               'shouted', 'whispered', 'answered', 'replied', 'spoke', 'laughed'},
    )
    
    PRONOUN_SIGNATURE = CategorySignature(
        name="pronoun",
        min_frequency=0.001,
        typical_positions=['start', 'middle'],
        position_weight=0.3,
        min_length=1,
        max_length=10,
        length_weight=0.3,
        seeds={'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
               'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'},
    )
    
    ENTITY_SIGNATURE = CategorySignature(
        name="entity",
        max_frequency=0.01,  # Entities are relatively rare
        max_document_frequency=0.3,  # Don't appear everywhere
        min_length=3,
        length_weight=0.2,
        # No seeds - entities are discovered from capitalization
    )
    
    def __init__(self):
        self.name = "EmergentClassifierGear"
        
        # Category signatures
        self.signatures: Dict[str, CategorySignature] = {
            'stopword': self.STOPWORD_SIGNATURE,
            'verb': self.VERB_SIGNATURE,
            'pronoun': self.PRONOUN_SIGNATURE,
            'entity': self.ENTITY_SIGNATURE,
        }
        
        # Learned word statistics
        self.word_stats: Dict[str, WordStats] = {}
        self.total_words = 0
        self.total_documents = 0
        
        # Cached classifications
        self._cache: Dict[str, Dict[str, float]] = {}
        self._cache_valid = False
    
    def add_signature(self, signature: CategorySignature):
        """Add a new category signature."""
        self.signatures[signature.name] = signature
        self._cache_valid = False
    
    def learn_from_text(self, text: str, document_id: str = None):
        """Learn word statistics from text."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Track document
        if document_id:
            self.total_documents += 1
            seen_in_doc = set()
        
        # Update word stats
        for i, word in enumerate(words):
            self.total_words += 1
            
            if word not in self.word_stats:
                self.word_stats[word] = WordStats(word=word)
            
            stats = self.word_stats[word]
            stats.count += 1
            stats.total_words = self.total_words
            stats.total_documents = self.total_documents
            
            # Track position (normalized 0-1)
            if len(words) > 1:
                position = i / (len(words) - 1)
                stats.position_sum += position
                stats.position_count += 1
            
            # Track document frequency
            if document_id and word not in seen_in_doc:
                stats.document_count += 1
                seen_in_doc.add(word)
        
        self._cache_valid = False
    
    def classify(self, word: str) -> Dict[str, float]:
        """
        Classify a word into categories.
        
        Returns dict of category -> confidence score (0-1).
        """
        word_lower = word.lower()
        
        # Check cache
        if self._cache_valid and word_lower in self._cache:
            return self._cache[word_lower]
        
        # Get or create stats
        stats = self.word_stats.get(word_lower, WordStats(word=word_lower))
        stats.total_words = max(self.total_words, 1)
        stats.total_documents = max(self.total_documents, 1)
        
        # Score against each signature
        scores = {}
        for name, signature in self.signatures.items():
            scores[name] = signature.score(word_lower, stats)
        
        # Cache result
        self._cache[word_lower] = scores
        
        return scores
    
    def is_category(self, word: str, category: str, threshold: float = 0.5) -> bool:
        """Check if a word belongs to a category."""
        scores = self.classify(word)
        return scores.get(category, 0) >= threshold
    
    def get_category(self, word: str, threshold: float = 0.5) -> Optional[str]:
        """Get the best matching category for a word."""
        scores = self.classify(word)
        best = max(scores.items(), key=lambda x: x[1])
        if best[1] >= threshold:
            return best[0]
        return None
    
    def get_words_in_category(self, category: str, threshold: float = 0.5) -> Set[str]:
        """Get all known words that belong to a category."""
        result = set()
        for word in self.word_stats:
            if self.is_category(word, category, threshold):
                result.add(word)
        return result
    
    def is_stopword(self, word: str) -> bool:
        """Convenience method for stopword detection."""
        return self.is_category(word, 'stopword', threshold=0.4)
    
    def is_verb(self, word: str) -> bool:
        """Convenience method for verb detection."""
        return self.is_category(word, 'verb', threshold=0.4)
    
    def is_pronoun(self, word: str) -> bool:
        """Convenience method for pronoun detection."""
        return self.is_category(word, 'pronoun', threshold=0.5)
    
    def is_entity(self, word: str, original_case: str = None) -> bool:
        """
        Convenience method for entity detection.
        
        Uses capitalization as additional signal if original_case provided.
        """
        if original_case and original_case[0].isupper():
            # Capitalized words get a boost for entity detection
            scores = self.classify(word)
            return scores.get('entity', 0) >= 0.3 or not self.is_stopword(word)
        return self.is_category(word, 'entity', threshold=0.5)
    
    def process_message(self, message: GearMessage) -> GearMessage:
        """Classify words in message. Implements GearProtocol."""
        words = re.findall(r'\b[a-zA-Z]+\b', message.content.lower())
        classifications = {}
        for word in set(words):
            classifications[word] = self.classify(word)
        return self.send(
            message.with_context('classifications', classifications),
            content=message.content
        )
    
    def forward(self, state: GearState) -> GearState:
        """Classify the entity in the state (legacy interface)."""
        word = state.entity
        if word:
            scores = self.classify(word)
            state.metadata['word_categories'] = scores
            state.metadata['primary_category'] = self.get_category(word)
        return state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        return {
            'total_words_seen': self.total_words,
            'unique_words': len(self.word_stats),
            'total_documents': self.total_documents,
            'categories': list(self.signatures.keys()),
        }


# =============================================================================
# PROTOCOL: Converting Hardcoded Concepts to Emergent Gears
# =============================================================================

"""
PROTOCOL FOR EMERGENT CONCEPT DISCOVERY

When you find yourself writing a hardcoded list like:
    stopwords = {'the', 'a', 'an', ...}
    verbs = {'said', 'went', 'came', ...}

Follow this protocol to make it emergent:

1. IDENTIFY THE STRUCTURAL SIGNAL
   Ask: "What makes these words different from others?"
   
   Examples:
   - Stopwords: Very frequent, appear everywhere, short
   - Verbs: Appear after subjects, have -ed/-ing forms
   - Entities: Capitalized, less frequent, specific contexts
   - Pronouns: Very frequent, specific positions, closed class

2. CREATE A SIGNATURE
   ```python
   my_signature = CategorySignature(
       name="my_category",
       min_frequency=0.01,  # Adjust based on signal
       positive_suffixes=['ed', 'ing'],  # Morphological patterns
       typical_positions=['middle'],  # Syntactic position
       seeds={'example1', 'example2'},  # Bootstrap examples
   )
   ```

3. ADD TO CLASSIFIER
   ```python
   classifier = EmergentClassifierGear()
   classifier.add_signature(my_signature)
   ```

4. LEARN FROM DATA
   ```python
   for document in corpus:
       classifier.learn_from_text(document, document_id=doc_id)
   ```

5. USE EMERGENTLY
   ```python
   if classifier.is_category(word, 'my_category'):
       # Word belongs to category
   ```

6. VALIDATE AND REFINE
   - Check precision/recall against known examples
   - Adjust signature parameters
   - Add more seeds if needed

BENEFITS:
- No hardcoded lists to maintain
- Adapts to different domains/languages
- Discovers new members automatically
- Transparent: you can inspect why a word was classified
"""


def create_custom_signature(
    name: str,
    description: str,
    seeds: Set[str] = None,
    frequency_range: Tuple[float, float] = (0, 1),
    length_range: Tuple[int, int] = (1, 50),
    suffixes: List[str] = None,
    positions: List[str] = None,
) -> CategorySignature:
    """
    Helper to create a custom category signature.
    
    Args:
        name: Category name
        description: Human-readable description
        seeds: Example words (for bootstrapping)
        frequency_range: (min, max) relative frequency
        length_range: (min, max) word length
        suffixes: Characteristic suffixes
        positions: Typical positions ('start', 'middle', 'end')
    
    Returns:
        CategorySignature ready to add to classifier
    """
    return CategorySignature(
        name=name,
        min_frequency=frequency_range[0],
        max_frequency=frequency_range[1],
        min_length=length_range[0],
        max_length=length_range[1],
        length_weight=0.3 if length_range != (1, 50) else 0,
        positive_suffixes=suffixes or [],
        suffix_weight=0.5 if suffixes else 0,
        typical_positions=positions or [],
        position_weight=0.3 if positions else 0,
        seeds=seeds or set(),
    )
