"""
KnowledgeSpace - HyperMapping-based Knowledge Storage

Replaces GeometricKnowledgeStore with a cleaner HyperMapping-based API.

The key insight: A knowledge store IS a HyperMapping where:
- Input = query text
- Output = concept/response text
- Position = geometric encoding of meaning

Design Principles (from Design 091):
- POSITION IS IDENTITY
- MOVEMENT IS LEARNING
- THE CRITICAL LINE IS THE HORIZON

Concepts start at the origin. Success moves them toward persistence.
Failure moves them toward pruning. The critical line (σ = 0.5) is the
natural boundary between persisting and fading concepts.

Author: Lesley Gushurst
License: GPLv3
"""

import re
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

# Import from hypermapping package
import sys
hypermapping_path = Path(__file__).parent.parent.parent / 'hypermapping'
if str(hypermapping_path) not in sys.path:
    sys.path.insert(0, str(hypermapping_path))

from hypermapping import HyperMapping, Mapping, MatchResult, TextEncoder, CRITICAL_LINE


@dataclass
class Concept:
    """
    A concept in the knowledge space.
    
    This is a thin wrapper around Mapping that provides
    concept-specific functionality.
    """
    words: Set[str]
    source: str = "unknown"
    text_snippets: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"concept_{time.time()}")
    created: float = field(default_factory=time.time)
    
    @property
    def text(self) -> str:
        """Primary text representation."""
        return self.text_snippets[0] if self.text_snippets else " ".join(self.words)


class KnowledgeSpace(HyperMapping):
    """
    Knowledge storage using HyperMapping.
    
    Replaces GeometricKnowledgeStore with a cleaner, more geometric API.
    
    Key differences from GeometricKnowledgeStore:
    - Uses HyperMapping's position-based matching
    - No separate Concept class - uses Mapping with metadata
    - Emergent stop word detection (geometric, not hardcoded)
    - Learning through position reinforcement
    
    Usage:
        space = KnowledgeSpace(name="chat_knowledge")
        
        # Add knowledge
        space.add_text("The capital of France is Paris", source="wikipedia")
        
        # Query
        results = space.query_text("What is the capital of France?")
        
        # Feedback
        space.use(results[0].mapping, success=True)
        
        # Persistence
        space.save("knowledge.json")
        space = KnowledgeSpace.load("knowledge.json")
    """
    
    def __init__(self, name: str = "knowledge", dims: int = 8):
        # Create TextEncoder for text-based knowledge
        encoder = TextEncoder(dims=dims)
        super().__init__(dims=dims, encoder=encoder, name=name)
        
        # Word frequency tracking for emergent stop word detection
        self._word_counts: Dict[str, int] = {}
        self._total_concepts: int = 0
        
        # Metadata
        self.created = datetime.now().isoformat()
        self.modified = datetime.now().isoformat()
    
    # -------------------------------------------------------------------------
    # Text Processing (Geometric)
    # -------------------------------------------------------------------------
    
    def extract_words(self, text: str) -> Set[str]:
        """
        Extract content words from text using geometric stop word detection.
        
        Geometric principle (from Design 062):
        - Content words have clear semantic roles (appear in specific contexts)
        - Stop words have NO semantic role (appear everywhere uniformly)
        - Detection is based on distribution, not a hardcoded list
        """
        # Tokenize: split on non-alphanumeric, preserve case for acronym detection
        raw_words = re.findall(r'\b[a-zA-Z]+\b', text)
        
        # Filter using geometric detection, lowercase for matching
        content_words = set()
        for w in raw_words:
            # Check if it's an acronym (all uppercase, 2+ chars)
            is_acronym = w.isupper() and len(w) >= 2
            w_lower = w.lower()
            
            if is_acronym or self._is_content_word(w_lower):
                content_words.add(w_lower)
        
        return content_words
    
    def _is_content_word(self, word: str) -> bool:
        """
        Geometric stop word detection.
        
        A word is a STOP word (not content) if:
        1. It's very short (< 2 chars) - structural, not semantic
        2. It appears in many concepts but adds no discriminative power
        
        This is geometric because it's based on the word's distribution
        across the concept space, not a hardcoded list.
        
        Note: Acronyms like "AI", "ML", "NLP" are kept (2+ chars, uppercase pattern)
        """
        # Single char words are structural
        if len(word) < 2:
            return False
        
        # Keep acronyms (all uppercase, 2+ chars)
        if word.upper() == word and len(word) >= 2:
            return True
        
        # Very short lowercase words are likely structural
        if len(word) < 3:
            return False
        
        # If we have concept data, use it for geometric detection
        if word in self._word_counts and self._total_concepts > 1:
            # Word appears in what fraction of concepts?
            coverage = self._word_counts[word] / self._total_concepts
            
            # High coverage (> 50%) = stop word (appears everywhere)
            # This threshold is the critical line (σ = 0.5)
            if coverage > CRITICAL_LINE:
                return False
        
        return True
    
    @staticmethod
    def word_overlap(words_a: Set[str], words_b: Set[str]) -> float:
        """
        Calculate Jaccard similarity between two word sets.
        
        Returns value in [0, 1] where 1 means identical sets.
        """
        if not words_a or not words_b:
            return 0.0
        
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        
        return intersection / union if union > 0 else 0.0
    
    # -------------------------------------------------------------------------
    # Knowledge Operations
    # -------------------------------------------------------------------------
    
    def add_text(self, text: str, source: str = "unknown",
                 reproject: bool = True) -> Mapping:
        """
        Add knowledge from text.
        
        The text is used as both input and output (self-mapping for concepts).
        Position is computed from the text content.
        
        Args:
            text: The knowledge text
            source: Source attribution
            reproject: Whether to reproject after adding
            
        Returns:
            The created Mapping
        """
        words = self.extract_words(text)
        
        # Update word counts for geometric stop word detection
        for word in words:
            self._word_counts[word] = self._word_counts.get(word, 0) + 1
        self._total_concepts += 1
        
        # Create mapping (text → text for concepts)
        mapping = self.map(text, text, metadata={
            'source': source,
            'words': list(words),
            'type': 'concept',
        })
        
        # Reproject if requested - use content word similarity
        if reproject and len(self) > 1:
            self.reproject(similarity_fn=self._content_word_similarity)
        
        self.modified = datetime.now().isoformat()
        return mapping
    
    def _content_word_similarity(self, a: Any, b: Any) -> float:
        """
        Compute similarity using content words only.
        
        This is the key to geometric matching - we compare content words,
        not full text. This gives much higher similarity for related concepts.
        """
        # Get words from mapping metadata if available
        if hasattr(a, 'metadata') and a.metadata and 'words' in a.metadata:
            words_a = set(a.metadata['words'])
        elif hasattr(a, 'input'):
            words_a = self.extract_words(str(a.input))
        else:
            words_a = self.extract_words(str(a))
        
        if hasattr(b, 'metadata') and b.metadata and 'words' in b.metadata:
            words_b = set(b.metadata['words'])
        elif hasattr(b, 'input'):
            words_b = self.extract_words(str(b.input))
        else:
            words_b = self.extract_words(str(b))
        
        return self.word_overlap(words_a, words_b)
    
    def query_text(self, text: str, top_k: int = 5,
                   min_similarity: float = 0.0) -> List[MatchResult]:
        """
        Find concepts most similar to the query text.
        
        Uses geometric position matching with content word similarity.
        
        Args:
            text: Query text
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of MatchResult, sorted by similarity descending
        """
        # Project query using content word similarity
        if hasattr(self, '_eigenvectors') and self._eigenvectors is not None:
            query_position = self.project_query(
                text, 
                similarity_fn=lambda q, m: self._content_word_similarity(q, m)
            )
        else:
            query_position = self.encoder.encode_input(text)
        
        # Find nearest by position
        results = self._query_by_position(query_position, top_k)
        
        # Filter by minimum similarity
        if min_similarity > 0:
            results = [r for r in results if r.similarity >= min_similarity]
        
        return results
    
    def use(self, mapping: Mapping, success: bool,
            strength: float = 0.1) -> None:
        """
        Record use of a mapping and reinforce based on success.
        
        This is THE learning operation:
        - Success: Move position outward (toward persistence)
        - Failure: Move position inward (toward pruning)
        
        Args:
            mapping: The mapping that was used
            success: Whether the use was successful
            strength: How much to reinforce
        """
        self.reinforce(mapping, success, strength)
        self.modified = datetime.now().isoformat()
    
    def get_concept(self, text: str) -> Optional[Mapping]:
        """Get a concept by its text."""
        for mapping in self._mappings:
            if mapping.input == text:
                return mapping
        return None
    
    # -------------------------------------------------------------------------
    # Compatibility with GeometricKnowledgeStore
    # -------------------------------------------------------------------------
    
    def add_from_text(self, text: str, source: str = "text") -> Mapping:
        """Alias for add_text (compatibility)."""
        return self.add_text(text, source)
    
    def get_persisting_concepts(self) -> List[Mapping]:
        """Alias for get_persisting (compatibility)."""
        return self.get_persisting()
    
    def get_fading_concepts(self) -> List[Mapping]:
        """Alias for get_fading (compatibility)."""
        return self.get_fading()
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = super().to_dict()
        data['type'] = 'KnowledgeSpace'
        data['word_counts'] = self._word_counts
        data['total_concepts'] = self._total_concepts
        data['created'] = self.created
        data['modified'] = self.modified
        return data
    
    def save(self, path: str) -> None:
        """Save to JSON file."""
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeSpace':
        """Deserialize from dictionary."""
        space = cls(
            name=data.get('name', 'knowledge'),
            dims=data.get('dims', 8)
        )
        
        # Load mappings
        for m_data in data.get('mappings', []):
            mapping = Mapping.from_dict(m_data)
            space._mappings.append(mapping)
        
        # Rebuild indices
        space._rebuild_indices()
        
        # Load word counts
        space._word_counts = data.get('word_counts', {})
        space._total_concepts = data.get('total_concepts', 0)
        
        # Load metadata
        space.created = data.get('created', datetime.now().isoformat())
        space.modified = data.get('modified', datetime.now().isoformat())
        
        # Load templates if present
        if 'templates' in data:
            space._templates = data['templates']
        
        return space
    
    @classmethod
    def load(cls, path: str) -> 'KnowledgeSpace':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------
    
    def merge(self, other: 'KnowledgeSpace') -> None:
        """
        Merge another KnowledgeSpace into this one.
        
        Concepts from other are added to this space.
        Duplicate texts are skipped.
        """
        existing_texts = {m.input for m in self._mappings}
        
        for mapping in other._mappings:
            if mapping.input not in existing_texts:
                # Copy mapping
                new_mapping = Mapping(
                    input=mapping.input,
                    output=mapping.output,
                    position=mapping.position.copy(),
                    metadata=mapping.metadata.copy(),
                    use_count=mapping.use_count,
                    success_count=mapping.success_count,
                    created=mapping.created,
                )
                self._mappings.append(new_mapping)
                
                # Update word counts
                words = mapping.metadata.get('words', [])
                for word in words:
                    self._word_counts[word] = self._word_counts.get(word, 0) + 1
                self._total_concepts += 1
        
        # Rebuild indices
        self._rebuild_indices()
        
        # Reproject
        if len(self) > 1:
            self.reproject()
        
        self.modified = datetime.now().isoformat()
    
    def __repr__(self) -> str:
        return f"KnowledgeSpace(name='{self.name}', concepts={len(self)}, persisting={len(self.get_persisting())})"
