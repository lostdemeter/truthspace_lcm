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

Similarity Formula (from Design 039):
- importance = phi_weight(A) × phi_weight(B) × spread × bidir
- phi_weight(X) = φ^(-rank(X))
- This is geometric: encoding and weighting are dual operations

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

# Golden ratio for φ-weighting
PHI = (1 + np.sqrt(5)) / 2


@dataclass
class Entity:
    """
    An entity (content word) with geometric importance data.
    
    From Design 039:
    - importance = phi_weight(A) × phi_weight(B) × spread × bidir
    - phi_weight(X) = φ^(-rank(X))
    """
    name: str
    frequency: int = 0
    rank: int = 0
    sources: Set[str] = field(default_factory=set)
    relationships: Dict[str, int] = field(default_factory=dict)  # entity -> count
    
    @property
    def spread(self) -> int:
        """How many sources mention this entity."""
        return len(self.sources)
    
    def bidir(self, other: 'Entity') -> float:
        """
        Bidirectional relationship strength with another entity.
        
        If A mentions B AND B mentions A, that's a strong signal.
        Returns geometric mean of bidirectional counts.
        """
        a_to_b = self.relationships.get(other.name, 0)
        b_to_a = other.relationships.get(self.name, 0)
        
        if a_to_b == 0 or b_to_a == 0:
            return 0.0
        
        return np.sqrt(a_to_b * b_to_a)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entity."""
        return {
            'name': self.name,
            'frequency': self.frequency,
            'rank': self.rank,
            'sources': list(self.sources),
            'relationships': self.relationships,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Deserialize entity."""
        return cls(
            name=data['name'],
            frequency=data.get('frequency', 0),
            rank=data.get('rank', 0),
            sources=set(data.get('sources', [])),
            relationships=data.get('relationships', {}),
        )


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
        
        # Entity tracking for φ^(-rank) importance (Design 039)
        self._entities: Dict[str, Entity] = {}
        self._ranks_computed: bool = False
        
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
        
        All text is lowercased - case is orthographic noise, not semantic.
        The geometry handles similarity through word co-occurrence.
        """
        # Tokenize and lowercase - case is not semantic information
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter using geometric detection
        return {w for w in words if self._is_content_word(w)}
    
    def _is_content_word(self, word: str) -> bool:
        """
        Fully geometric stop word detection (Emergent Gear Pattern).
        
        NO HARDCODED STOP WORD LIST. Detection is purely emergent:
        
        1. Short words (< 3 chars) are structural - length is a property
        2. High coverage (> critical line) = appears everywhere = structural
        
        Key insight: Structural words appear across MANY topics uniformly.
        Domain words like "python" may be frequent but are topic-specific.
        
        The geometric signal is COVERAGE (fraction of concepts containing word),
        not raw frequency. A word appearing 10 times in 2 concepts about Python
        is content. A word appearing 10 times across 10 different topics is structural.
        
        This follows the Emergent Gear Pattern (Design 086):
        - STRUCTURE: Coverage is the geometric property
        - BOOTSTRAP: Initial concepts establish coverage distribution
        - MATCH: Coverage > critical line → structural
        - LEARN: As more concepts added, coverage distribution refines
        """
        # Short words are structural (length is a property, not a list)
        if len(word) < 3:
            return False
        
        # Geometric detection via coverage
        # Coverage = fraction of concepts containing this word
        # High coverage (> critical line) = appears everywhere = structural
        if word in self._word_counts and self._total_concepts > 1:
            coverage = self._word_counts[word] / self._total_concepts
            
            # Critical line (σ = 0.5) is the threshold
            # Words in > 50% of concepts are structural scaffolding
            if coverage > CRITICAL_LINE:
                return False
        
        return True
    
    # -------------------------------------------------------------------------
    # φ^(-rank) Importance Formula (Design 039)
    # -------------------------------------------------------------------------
    
    def _compute_ranks(self) -> None:
        """Compute ranks based on frequency (most frequent = rank 1)."""
        sorted_entities = sorted(
            self._entities.values(),
            key=lambda e: -e.frequency
        )
        for rank, entity in enumerate(sorted_entities, 1):
            entity.rank = rank
        self._ranks_computed = True
    
    def phi_weight(self, entity_name: str) -> float:
        """
        φ-based weighting for an entity using normalized rank.
        
        Instead of φ^(-rank) which decays too fast for large vocabularies,
        we use φ^(-log(rank)) which gives a more gradual decay.
        
        This is equivalent to rank^(-log(φ)) ≈ rank^(-0.48), a power law.
        """
        if not self._ranks_computed:
            self._compute_ranks()
        
        entity = self._entities.get(entity_name)
        if not entity:
            return 0.0
        
        # Use log(rank) to slow down the decay
        # φ^(-log(rank)) = rank^(-log(φ)) ≈ rank^(-0.48)
        log_rank = np.log1p(entity.rank)  # log(1 + rank) to handle rank=0
        return PHI ** (-log_rank)
    
    def entity_importance(self, entity_a: str, entity_b: str) -> float:
        """
        Compute importance of relationship between two entities.
        
        From Design 039:
        importance = phi_weight(A) × phi_weight(B) × spread × bidir
        
        For same-entity matching (A == B), we use:
        importance = phi_weight(A)² × spread
        
        For related entities (co-occur in concepts), we use the full formula.
        For unrelated entities, importance is 0.
        """
        if entity_a not in self._entities or entity_b not in self._entities:
            return 0.0
        
        ent_a = self._entities[entity_a]
        ent_b = self._entities[entity_b]
        
        phi_a = self.phi_weight(entity_a)
        phi_b = self.phi_weight(entity_b)
        
        # Same entity - this is the primary matching signal
        if entity_a == entity_b:
            # importance = φ² × spread
            return phi_a * phi_a * ent_a.spread
        
        # Different entities - check for relationship
        # Relationship exists if they co-occur in any concept
        a_to_b = ent_a.relationships.get(entity_b, 0)
        b_to_a = ent_b.relationships.get(entity_a, 0)
        
        if a_to_b == 0 and b_to_a == 0:
            return 0.0
        
        # Spread: geometric mean of source counts
        spread = np.sqrt(ent_a.spread * ent_b.spread)
        
        # Relationship strength: geometric mean of co-occurrence counts
        # This is more lenient than strict bidirectionality
        relationship = np.sqrt(max(a_to_b, 1) * max(b_to_a, 1)) if (a_to_b > 0 or b_to_a > 0) else 0.0
        
        return phi_a * phi_b * spread * relationship
    
    def text_importance(self, words_a: Set[str], words_b: Set[str]) -> float:
        """
        Compute importance between two texts using φ^(-rank) formula.
        
        Sums importance of all entity pairs between the two texts.
        This replaces Jaccard word overlap with geometric importance.
        """
        if not words_a or not words_b:
            return 0.0
        
        if not self._ranks_computed:
            self._compute_ranks()
        
        total_importance = 0.0
        
        for a in words_a:
            for b in words_b:
                total_importance += self.entity_importance(a, b)
        
        # Normalize by geometric mean of word counts
        normalizer = np.sqrt(len(words_a) * len(words_b))
        return total_importance / normalizer if normalizer > 0 else 0.0
    
    @staticmethod
    def word_overlap(words_a: Set[str], words_b: Set[str]) -> float:
        """
        Calculate Jaccard similarity between two word sets.
        
        DEPRECATED: Use text_importance() instead for geometric similarity.
        Kept for backward compatibility.
        
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
        
        Entity tracking (Design 039):
        - Updates entity frequency and sources
        - Tracks entity relationships (co-occurrence in same concept)
        - Invalidates rank cache for recomputation
        
        Args:
            text: The knowledge text
            source: Source attribution
            reproject: Whether to reproject after adding
            
        Returns:
            The created Mapping
        """
        words = self.extract_words(text)
        word_list = list(words)
        
        # Update word counts for geometric stop word detection
        for word in words:
            self._word_counts[word] = self._word_counts.get(word, 0) + 1
        self._total_concepts += 1
        
        # Update entity tracking for φ^(-rank) importance
        for i, word in enumerate(word_list):
            if word not in self._entities:
                self._entities[word] = Entity(name=word)
            
            entity = self._entities[word]
            entity.frequency += 1
            entity.sources.add(source)
            
            # Track relationships (co-occurrence in same concept)
            for j, other in enumerate(word_list):
                if i != j:
                    entity.relationships[other] = entity.relationships.get(other, 0) + 1
        
        # Invalidate rank cache
        self._ranks_computed = False
        
        # Create mapping (text → text for concepts)
        mapping = self.map(text, text, metadata={
            'source': source,
            'words': word_list,
            'type': 'concept',
        })
        
        # Reproject if requested - use φ-importance similarity
        if reproject and len(self) > 1:
            self.reproject(similarity_fn=self._phi_importance_similarity)
        
        self.modified = datetime.now().isoformat()
        return mapping
    
    def _extract_words_from_item(self, item: Any) -> Set[str]:
        """Extract words from a mapping, text, or other item."""
        if hasattr(item, 'metadata') and item.metadata and 'words' in item.metadata:
            return set(item.metadata['words'])
        elif hasattr(item, 'input'):
            return self.extract_words(str(item.input))
        else:
            return self.extract_words(str(item))
    
    def _phi_importance_similarity(self, a: Any, b: Any) -> float:
        """
        Compute similarity using φ^(-rank) importance formula.
        
        From Design 039:
        importance = phi_weight(A) × phi_weight(B) × spread × bidir
        
        This replaces Jaccard word overlap with geometric importance.
        """
        words_a = self._extract_words_from_item(a)
        words_b = self._extract_words_from_item(b)
        
        return self.text_importance(words_a, words_b)
    
    def _content_word_similarity(self, a: Any, b: Any) -> float:
        """
        Compute similarity using content words only.
        
        DEPRECATED: Use _phi_importance_similarity instead.
        Kept for backward compatibility.
        """
        words_a = self._extract_words_from_item(a)
        words_b = self._extract_words_from_item(b)
        
        return self.word_overlap(words_a, words_b)
    
    def query_text(self, text: str, top_k: int = 5,
                   min_similarity: float = 0.0) -> List[MatchResult]:
        """
        Find concepts most similar to the query text.
        
        Uses φ^(-rank) importance formula for direct matching.
        
        Note: We use direct importance calculation rather than position-based
        matching because the eigenspace projection compresses differences.
        The φ-importance formula provides better discrimination.
        
        Args:
            text: Query text
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of MatchResult, sorted by similarity descending
        """
        query_words = self.extract_words(text)
        
        # Direct importance calculation for each concept
        results = []
        for mapping in self._mappings:
            concept_words = set(mapping.metadata.get('words', []))
            importance = self.text_importance(query_words, concept_words)
            
            results.append(MatchResult(
                mapping=mapping,
                similarity=importance,
            ))
        
        # Sort by importance descending
        results.sort(key=lambda r: -r.similarity)
        
        # Filter by minimum similarity
        if min_similarity > 0:
            results = [r for r in results if r.similarity >= min_similarity]
        
        return results[:top_k]
    
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
        
        # Serialize entities for φ^(-rank) importance
        data['entities'] = {
            name: entity.to_dict() 
            for name, entity in self._entities.items()
        }
        
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
        
        # Load entities for φ^(-rank) importance
        entities_data = data.get('entities', {})
        space._entities = {
            name: Entity.from_dict(ent_data)
            for name, ent_data in entities_data.items()
        }
        space._ranks_computed = False
        
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
        Entity data is merged for φ^(-rank) importance.
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
        
        # Merge entity data for φ^(-rank) importance
        for name, other_entity in other._entities.items():
            if name not in self._entities:
                self._entities[name] = Entity(name=name)
            
            entity = self._entities[name]
            entity.frequency += other_entity.frequency
            entity.sources.update(other_entity.sources)
            
            # Merge relationships
            for rel_name, count in other_entity.relationships.items():
                entity.relationships[rel_name] = entity.relationships.get(rel_name, 0) + count
        
        # Invalidate rank cache
        self._ranks_computed = False
        
        # Rebuild indices
        self._rebuild_indices()
        
        # Reproject with φ-importance similarity
        if len(self) > 1:
            self.reproject(similarity_fn=self._phi_importance_similarity)
        
        self.modified = datetime.now().isoformat()
    
    def __repr__(self) -> str:
        return f"KnowledgeSpace(name='{self.name}', concepts={len(self)}, persisting={len(self.get_persisting())})"
