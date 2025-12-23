#!/usr/bin/env python3
"""
Holographic Generator: Interference-Based Text Generation

This module implements holographic generation using the principle:
- Multiple source texts act as reference beams
- Query acts as object beam
- Interference pattern determines output content
- Constructive interference = include, destructive = exclude

The key insight from Feynman's path integral:
- Each path (source text) contributes with a phase
- Phases that align reinforce (constructive)
- Phases that oppose cancel (destructive)

Author: Lesley Gushurst
License: GPLv3
"""

import re
import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# WORD CATEGORIES FOR PHASE ASSIGNMENT
# =============================================================================

# Content words: phase = 0 (real axis, positive)
CONTENT_WORDS = {
    # Entities
    'holmes', 'watson', 'darcy', 'elizabeth', 'jane', 'bingley',
    'moriarty', 'lestrade', 'wickham', 'charlotte',
    # Objects
    'evidence', 'clue', 'letter', 'book', 'door', 'room', 'house',
    'garden', 'carriage', 'horse', 'hat', 'coat',
    # Concepts
    'detective', 'gentleman', 'lady', 'friend', 'villain', 'doctor',
    'mystery', 'crime', 'love', 'pride', 'prejudice',
}

# Action words: phase = π/2 (imaginary axis)
ACTION_WORDS = {
    'investigated', 'examined', 'studied', 'observed', 'noticed',
    'spoke', 'said', 'told', 'asked', 'replied', 'answered',
    'walked', 'ran', 'went', 'came', 'arrived', 'left',
    'loved', 'hated', 'feared', 'admired', 'challenged',
    'thought', 'believed', 'considered', 'wondered', 'realized',
}

# Modifier words: phase = π (real axis, negative - cancels noise)
MODIFIER_WORDS = {
    'the', 'a', 'an', 'of', 'to', 'in', 'on', 'at', 'by', 'for',
    'with', 'from', 'as', 'is', 'was', 'were', 'be', 'been',
    'very', 'quite', 'rather', 'somewhat', 'indeed', 'certainly',
    'he', 'she', 'it', 'they', 'him', 'her', 'his', 'their',
}

# Quality words: phase = 3π/2 (negative imaginary)
QUALITY_WORDS = {
    'brilliant', 'clever', 'proud', 'witty', 'kind', 'loyal',
    'cunning', 'mysterious', 'beautiful', 'handsome', 'brave',
    'careful', 'carefully', 'quickly', 'slowly', 'quietly',
}


def get_word_phase(word: str) -> float:
    """Get phase for a word based on its category."""
    word_lower = word.lower()
    
    if word_lower in CONTENT_WORDS:
        return 0.0  # Real positive
    elif word_lower in ACTION_WORDS:
        return np.pi / 2  # Imaginary positive
    elif word_lower in MODIFIER_WORDS:
        return np.pi  # Real negative (cancels)
    elif word_lower in QUALITY_WORDS:
        return 3 * np.pi / 2  # Imaginary negative
    else:
        # Unknown words: hash to a phase
        return (hash(word_lower) % 1000) / 1000 * 2 * np.pi


def get_word_magnitude(word: str, idf: Dict[str, float] = None) -> float:
    """Get magnitude for a word (importance weighting)."""
    word_lower = word.lower()
    
    # Use IDF if available
    if idf and word_lower in idf:
        return idf[word_lower]
    
    # Default: content words are more important
    if word_lower in CONTENT_WORDS:
        return 2.0
    elif word_lower in ACTION_WORDS:
        return 1.5
    elif word_lower in QUALITY_WORDS:
        return 1.2
    elif word_lower in MODIFIER_WORDS:
        return 0.5
    else:
        return 1.0


# =============================================================================
# HOLOGRAPHIC GENERATOR
# =============================================================================

@dataclass
class InterferencePattern:
    """Result of holographic interference."""
    word_magnitudes: Dict[str, float] = field(default_factory=dict)
    word_phases: Dict[str, float] = field(default_factory=dict)
    source_count: int = 0
    
    def get_constructive_words(self, threshold: float = 1.5) -> List[str]:
        """Get words with constructive interference (high magnitude)."""
        return [w for w, m in self.word_magnitudes.items() if m >= threshold]
    
    def get_top_words(self, k: int = 10) -> List[Tuple[str, float]]:
        """Get top k words by magnitude."""
        sorted_words = sorted(self.word_magnitudes.items(), key=lambda x: -x[1])
        return sorted_words[:k]


class HolographicGenerator:
    """
    Generate text using holographic interference.
    
    Multiple source texts interfere to produce output:
    - Constructive interference: common concepts reinforced
    - Destructive interference: noise cancelled
    """
    
    def __init__(self, knowledge=None):
        """
        Initialize the holographic generator.
        
        Args:
            knowledge: Optional ConceptKnowledge for retrieving sources
        """
        self.knowledge = knowledge
        self.idf = {}  # Word importance weights
        
        if knowledge:
            self._compute_idf()
    
    def _compute_idf(self):
        """Compute IDF weights from corpus."""
        if not self.knowledge or not self.knowledge.frames:
            return
        
        # Count document frequency
        doc_freq = Counter()
        total_docs = len(self.knowledge.frames)
        
        for frame in self.knowledge.frames:
            text = frame.get('text', '')
            words = set(self._tokenize(text))
            for word in words:
                doc_freq[word] += 1
        
        # Compute IDF
        for word, freq in doc_freq.items():
            self.idf[word] = np.log(total_docs / (freq + 1)) + 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def encode_text(self, text: str) -> Dict[str, complex]:
        """
        Encode text as complex vector.
        
        Each word becomes a complex number:
        - Magnitude = importance (IDF-weighted)
        - Phase = category (content/action/modifier/quality)
        """
        words = self._tokenize(text)
        vector = {}
        
        for word in words:
            magnitude = get_word_magnitude(word, self.idf)
            phase = get_word_phase(word)
            
            # Complex number: magnitude * e^(i*phase)
            value = magnitude * np.exp(1j * phase)
            
            if word in vector:
                vector[word] += value  # Accumulate
            else:
                vector[word] = value
        
        return vector
    
    def interfere(self, sources: List[str]) -> InterferencePattern:
        """
        Compute interference pattern from multiple sources.
        
        Args:
            sources: List of source texts
        
        Returns:
            InterferencePattern with word magnitudes and phases
        """
        pattern = InterferencePattern(source_count=len(sources))
        
        if not sources:
            return pattern
        
        # Encode all sources
        encoded = [self.encode_text(s) for s in sources]
        
        # Sum complex vectors (interference)
        interference: Dict[str, complex] = {}
        for vec in encoded:
            for word, value in vec.items():
                if word in interference:
                    interference[word] += value
                else:
                    interference[word] = value
        
        # Extract magnitudes and phases
        for word, value in interference.items():
            pattern.word_magnitudes[word] = abs(value)
            pattern.word_phases[word] = np.angle(value)
        
        return pattern
    
    def generate(self, query: str, sources: List[str] = None, 
                 entity: str = None, max_words: int = 20,
                 learnable=None) -> str:
        """
        Generate text using holographic interference.
        
        Args:
            query: The query/question
            sources: Optional list of source texts (retrieved if not provided)
            entity: Optional entity to focus on
            max_words: Maximum words in output
            learnable: Optional LearnableStructure for enhanced generation
        
        Returns:
            Generated text
        """
        # If we have a learned profile, use it for better generation
        if learnable and entity:
            profile = learnable.get_profile(entity.lower())
            if profile.role or profile.qualities or profile.actions:
                return learnable.generate(entity, self._get_source_name(entity))
        
        # Retrieve sources if not provided
        if sources is None and self.knowledge and entity:
            sources = self._retrieve_sources(entity, k=5)
        
        if not sources:
            return f"No sources found for generation."
        
        # Compute interference
        pattern = self.interfere(sources)
        
        # Get constructive words (high magnitude)
        # Threshold scales with number of sources
        threshold = len(sources) * 0.8
        top_words = pattern.get_top_words(k=max_words)
        
        # Filter by threshold and category - also check if word is a known entity
        content = []
        actions = []
        qualities = []
        
        # Get known entities from knowledge base
        known_entities = set()
        if self.knowledge:
            known_entities = set(self.knowledge.entities.keys())
        
        for word, magnitude in top_words:
            if magnitude < threshold * 0.3:
                continue
            
            word_lower = word.lower()
            
            # Check if it's a known entity
            if word_lower in known_entities and word_lower != entity.lower():
                content.append(word)
            elif word_lower in CONTENT_WORDS:
                content.append(word)
            elif word_lower in ACTION_WORDS:
                actions.append(word)
            elif word_lower in QUALITY_WORDS:
                qualities.append(word)
        
        # Assemble output
        return self._assemble_output(entity, content, actions, qualities, query)
    
    def _get_source_name(self, entity: str) -> str:
        """Get the source name for an entity."""
        if not self.knowledge:
            return "the story"
        
        entity_info = self.knowledge.entities.get(entity.lower(), {})
        return entity_info.get('source', 'the story')
    
    def _retrieve_sources(self, entity: str, k: int = 5) -> List[str]:
        """Retrieve source texts for an entity."""
        if not self.knowledge:
            return []
        
        frames = self.knowledge.query_by_entity(entity, k=k)
        sources = []
        
        for frame in frames:
            text = frame.get('text', '')
            if text and len(text) > 10:
                sources.append(text)
        
        return sources
    
    def _assemble_output(self, entity: str, content: List[str], 
                         actions: List[str], qualities: List[str],
                         query: str) -> str:
        """Assemble output from interference results."""
        parts = []
        
        # Start with entity
        if entity:
            parts.append(entity.title())
            parts.append('is')
        
        # Add qualities
        if qualities:
            parts.append('a')
            parts.append(qualities[0])
        else:
            parts.append('a')
        
        # Add content (role/type)
        if content:
            # Filter out the entity itself
            other_content = [c for c in content if c.lower() != entity.lower()]
            if other_content:
                parts.append(other_content[0])
            else:
                parts.append('character')
        else:
            parts.append('character')
        
        # Add related entities
        related = [c for c in content if c.lower() != entity.lower() and c.lower() in CONTENT_WORDS]
        if len(related) > 1:
            parts.append(f'associated with {related[1]}')
        
        # Add actions
        if actions:
            parts.append(f'who {actions[0]}')
        
        # Build sentence
        if parts:
            sentence = ' '.join(parts)
            # Clean up
            sentence = re.sub(r'\s+', ' ', sentence)
            if not sentence.endswith('.'):
                sentence += '.'
            return sentence
        
        return f"Generated content about {entity}."
    
    def generate_from_frames(self, frames: List[dict], entity: str = None) -> str:
        """Generate from concept frames directly."""
        sources = [f.get('text', '') for f in frames if f.get('text')]
        return self.generate("", sources=sources, entity=entity)


# =============================================================================
# TEST
# =============================================================================

def test_holographic_generator():
    """Test the holographic generator."""
    print("=== HolographicGenerator Test ===\n")
    
    # Create generator without knowledge base
    gen = HolographicGenerator()
    
    # Test sources
    sources = [
        "Holmes examined the evidence carefully.",
        "The detective studied the mysterious clue.",
        "He observed the footprints with great interest.",
        "Holmes noticed something unusual about the letter.",
    ]
    
    print("Sources:")
    for s in sources:
        print(f"  - {s}")
    print()
    
    # Compute interference
    pattern = gen.interfere(sources)
    
    print("Interference pattern (top words):")
    for word, mag in pattern.get_top_words(10):
        phase = pattern.word_phases[word]
        phase_deg = np.degrees(phase)
        print(f"  {word:15} mag={mag:.2f}  phase={phase_deg:+.0f}°")
    print()
    
    # Generate
    output = gen.generate("Who is Holmes?", sources=sources, entity="holmes")
    print(f"Generated: {output}")
    print()
    
    # Test with different sources
    sources2 = [
        "Darcy is a proud gentleman from Pemberley.",
        "The proud Mr. Darcy loved Elizabeth.",
        "Darcy challenged his own pride for love.",
    ]
    
    print("Sources (Darcy):")
    for s in sources2:
        print(f"  - {s}")
    print()
    
    output2 = gen.generate("Who is Darcy?", sources=sources2, entity="darcy")
    print(f"Generated: {output2}")


if __name__ == '__main__':
    test_holographic_generator()
