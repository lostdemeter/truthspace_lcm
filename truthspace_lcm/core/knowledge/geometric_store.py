"""
GeometricKnowledgeStore - Persistent Geometric Knowledge Storage

The core insight: geometry IS the knowledge. We persist geometry directly,
not text that needs to be reconstructed into geometry.

This store manages:
1. Concepts (atomic units of knowledge)
2. Similarity matrix (word overlap between concepts)
3. Positions (derived from similarity via eigendecomposition)
4. Two-tier persistence (temporary → permanent via promotion)

Key Operations:
- add(): Add a concept, update geometry incrementally
- query(): Find nearest concepts by word overlap
- promote(): Move concept from temporary to permanent
- save()/load(): Persist to/from JSON

Design Principles (from ENCODE = DECODE):
- The space is conformally symmetric
- What works one direction must work the other
- Structure IS information

Author: Lesley Gushurst
License: GPLv3
"""

import json
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

from .concept import Concept, ConceptLevel


# Common stop words to filter out when extracting content words
STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
    'if', 'or', 'because', 'until', 'while', 'this', 'that', 'these',
    'those', 'what', 'which', 'who', 'whom', 'it', 'its', 'i', 'me', 'my',
    'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they',
    'them', 'their', 'about', 'also', 'any', 'both', 'but', 'even', 'get',
    'got', 'like', 'make', 'made', 'many', 'much', 'new', 'now', 'one',
    'out', 'over', 'say', 'said', 'see', 'take', 'time', 'up', 'use',
    'way', 'well', 'work', 'year', 'years', 'first', 'last', 'long',
    'great', 'little', 'own', 'old', 'right', 'big', 'high', 'different',
    'small', 'large', 'next', 'early', 'young', 'important', 'few', 'public',
    'bad', 'same', 'able',
}


@dataclass
class GeometricKnowledgeStore:
    """
    Persistent storage for geometric knowledge.
    
    The store maintains:
    - A list of concepts
    - A similarity matrix (word overlap)
    - Positions derived from the similarity matrix
    
    Attributes:
        name: Human-readable name for this store
        dims: Number of dimensions for position vectors
        concepts: List of all concepts
        similarity_matrix: N×N matrix of word overlap similarities
        positions: N×dims matrix of concept positions
        tier: 'temporary' or 'permanent'
    """
    
    name: str = "default"
    dims: int = 12
    tier: str = "temporary"
    
    # Concepts
    concepts: List[Concept] = field(default_factory=list)
    
    # Geometry (computed from concepts)
    similarity_matrix: Optional[np.ndarray] = None
    positions: Optional[np.ndarray] = None
    
    # Metadata
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Index for fast lookup
    _id_to_index: Dict[str, int] = field(default_factory=dict)
    _word_to_concepts: Dict[str, Set[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize indices after creation."""
        self._rebuild_indices()
    
    def _rebuild_indices(self) -> None:
        """Rebuild lookup indices from concepts."""
        self._id_to_index = {c.id: i for i, c in enumerate(self.concepts)}
        self._word_to_concepts = {}
        for concept in self.concepts:
            for word in concept.words:
                if word not in self._word_to_concepts:
                    self._word_to_concepts[word] = set()
                self._word_to_concepts[word].add(concept.id)
    
    @staticmethod
    def extract_words(text: str) -> Set[str]:
        """
        Extract content words from text.
        
        Filters out stop words and short words.
        Returns lowercase words.
        """
        # Tokenize: split on non-alphanumeric
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter stop words and short words
        content_words = {
            w for w in words 
            if w not in STOP_WORDS and len(w) > 2
        }
        
        return content_words
    
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
    
    def add(self, concept: Concept, reproject: bool = True) -> None:
        """
        Add a concept to the store.
        
        Args:
            concept: The concept to add
            reproject: Whether to recompute positions (default True)
        """
        # Check for duplicate ID
        if concept.id in self._id_to_index:
            # Update existing concept
            idx = self._id_to_index[concept.id]
            self.concepts[idx] = concept
        else:
            # Add new concept
            concept.position_index = len(self.concepts)
            self.concepts.append(concept)
            self._id_to_index[concept.id] = concept.position_index
        
        # Update word index
        for word in concept.words:
            if word not in self._word_to_concepts:
                self._word_to_concepts[word] = set()
            self._word_to_concepts[word].add(concept.id)
        
        # Update geometry
        if reproject:
            self._update_geometry_incremental(concept)
        
        self.modified = datetime.now().isoformat()
    
    def add_from_text(self, text: str, source: str = "text", 
                      temporary: bool = True) -> Concept:
        """
        Create and add a concept from text.
        
        Args:
            text: The text to create a concept from
            source: Source attribution
            temporary: Whether this is a temporary concept
            
        Returns:
            The created concept
        """
        words = self.extract_words(text)
        
        concept = Concept(
            words=words,
            source=source,
            temporary=temporary,
            text_snippets=[text],
        )
        
        self.add(concept)
        return concept
    
    def remove(self, concept_id: str) -> bool:
        """
        Remove a concept from the store.
        
        Args:
            concept_id: ID of the concept to remove
            
        Returns:
            True if removed, False if not found
        """
        if concept_id not in self._id_to_index:
            return False
        
        idx = self._id_to_index[concept_id]
        concept = self.concepts[idx]
        
        # Remove from word index
        for word in concept.words:
            if word in self._word_to_concepts:
                self._word_to_concepts[word].discard(concept_id)
        
        # Remove from concepts list
        del self.concepts[idx]
        
        # Rebuild indices (positions shifted)
        self._rebuild_indices()
        
        # Reproject geometry
        self._reproject()
        
        self.modified = datetime.now().isoformat()
        return True
    
    def get(self, concept_id: str) -> Optional[Concept]:
        """Get a concept by ID."""
        if concept_id not in self._id_to_index:
            return None
        return self.concepts[self._id_to_index[concept_id]]
    
    def query(self, text: str, top_k: int = 5, 
              min_similarity: float = 0.0) -> List[Tuple[Concept, float]]:
        """
        Find concepts most similar to the query text.
        
        Uses word overlap (Jaccard similarity) for matching.
        
        Args:
            text: Query text
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (concept, similarity) tuples, sorted by similarity descending
        """
        query_words = self.extract_words(text)
        
        if not query_words:
            return []
        
        # Calculate similarity to all concepts
        similarities = []
        for concept in self.concepts:
            sim = self.word_overlap(query_words, concept.words)
            if sim >= min_similarity:
                similarities.append((concept, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def query_by_position(self, position: np.ndarray, 
                          top_k: int = 5) -> List[Tuple[Concept, float]]:
        """
        Find concepts nearest to a position in geometric space.
        
        Args:
            position: Query position vector
            top_k: Maximum number of results
            
        Returns:
            List of (concept, distance) tuples, sorted by distance ascending
        """
        if self.positions is None or len(self.positions) == 0:
            return []
        
        # Calculate Euclidean distances
        distances = np.linalg.norm(self.positions - position, axis=1)
        
        # Get top-k nearest
        indices = np.argsort(distances)[:top_k]
        
        return [(self.concepts[i], distances[i]) for i in indices]
    
    def _update_geometry_incremental(self, new_concept: Concept) -> None:
        """
        Update geometry incrementally when adding a concept.
        
        Instead of full eigendecomposition, we approximate the new position
        as a weighted average of existing positions based on similarity.
        """
        n = len(self.concepts)
        
        if n == 1:
            # First concept - initialize matrices
            self.similarity_matrix = np.array([[1.0]])
            self.positions = np.zeros((1, self.dims))
            new_concept.quaternion = (1.0, 0.0, 0.0, 0.0)
            return
        
        # Compute similarity to all existing concepts
        new_similarities = np.array([
            self.word_overlap(new_concept.words, c.words)
            for c in self.concepts[:-1]  # Exclude the new concept itself
        ])
        
        # Extend similarity matrix
        new_row = np.append(new_similarities, 1.0)  # Self-similarity = 1
        
        # Expand matrix
        old_matrix = self.similarity_matrix
        self.similarity_matrix = np.zeros((n, n))
        self.similarity_matrix[:n-1, :n-1] = old_matrix
        self.similarity_matrix[n-1, :] = new_row
        self.similarity_matrix[:, n-1] = new_row
        
        # Approximate new position via weighted average
        if np.sum(new_similarities) > 0:
            weights = new_similarities / np.sum(new_similarities)
            new_position = weights @ self.positions[:n-1]
        else:
            # No overlap - place at origin (will be refined on reproject)
            new_position = np.zeros(self.dims)
        
        # Extend positions matrix
        self.positions = np.vstack([self.positions, new_position])
        
        # Update concept's quaternion from position
        # Use first 4 dimensions as quaternion (normalized)
        q = new_position[:4] if self.dims >= 4 else np.pad(new_position, (0, 4 - self.dims))
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            q = q / norm
        else:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        new_concept.quaternion = tuple(q)
    
    def _reproject(self) -> None:
        """
        Full reprojection of all positions from similarity matrix.
        
        Uses eigendecomposition to find positions such that
        dot(positions[i], positions[j]) ≈ similarity_matrix[i,j]
        """
        n = len(self.concepts)
        
        if n == 0:
            self.similarity_matrix = None
            self.positions = None
            return
        
        # Compute full similarity matrix
        self.similarity_matrix = np.zeros((n, n))
        for i, ci in enumerate(self.concepts):
            for j, cj in enumerate(self.concepts):
                self.similarity_matrix[i, j] = self.word_overlap(ci.words, cj.words)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(self.similarity_matrix)
        
        # Take top dims eigenvectors, scaled by sqrt(eigenvalue)
        # Sort by eigenvalue descending
        idx = np.argsort(eigenvalues)[::-1][:self.dims]
        
        # Handle negative eigenvalues (use absolute value)
        selected_eigenvalues = np.abs(eigenvalues[idx])
        selected_eigenvectors = eigenvectors[:, idx]
        
        self.positions = selected_eigenvectors * np.sqrt(selected_eigenvalues)
        
        # Update concept quaternions
        for i, concept in enumerate(self.concepts):
            concept.position_index = i
            q = self.positions[i, :4] if self.dims >= 4 else np.pad(self.positions[i], (0, 4 - self.dims))
            norm = np.linalg.norm(q)
            if norm > 1e-10:
                q = q / norm
            else:
                q = np.array([1.0, 0.0, 0.0, 0.0])
            concept.quaternion = tuple(q)
    
    def promote_qualifying(self) -> List[str]:
        """
        Promote all concepts that qualify for promotion.
        
        Returns:
            List of promoted concept IDs
        """
        promoted = []
        for concept in self.concepts:
            if concept.temporary and concept.qualifies_for_promotion:
                concept.promote()
                promoted.append(concept.id)
        
        if promoted:
            self.modified = datetime.now().isoformat()
        
        return promoted
    
    def get_temporary_concepts(self) -> List[Concept]:
        """Get all temporary concepts."""
        return [c for c in self.concepts if c.temporary]
    
    def get_permanent_concepts(self) -> List[Concept]:
        """Get all permanent concepts."""
        return [c for c in self.concepts if not c.temporary]
    
    def clear_temporary(self) -> int:
        """
        Remove all temporary concepts.
        
        Returns:
            Number of concepts removed
        """
        temp_ids = [c.id for c in self.concepts if c.temporary]
        for cid in temp_ids:
            self.remove(cid)
        return len(temp_ids)
    
    def merge(self, other: 'GeometricKnowledgeStore', 
              conflict_resolution: str = 'newer') -> int:
        """
        Merge another store into this one.
        
        Args:
            other: The store to merge from
            conflict_resolution: How to handle conflicts
                - 'newer': Keep the newer concept (by modified date)
                - 'higher_confidence': Keep the one with higher success rate
                - 'merge_words': Combine words from both
                
        Returns:
            Number of concepts added/updated
        """
        count = 0
        
        for other_concept in other.concepts:
            existing = self.get(other_concept.id)
            
            if existing is None:
                # New concept - add it
                self.add(other_concept, reproject=False)
                count += 1
            else:
                # Conflict - resolve based on strategy
                if conflict_resolution == 'newer':
                    if other_concept.modified > existing.modified:
                        self.add(other_concept, reproject=False)
                        count += 1
                elif conflict_resolution == 'higher_confidence':
                    if other_concept.success_rate > existing.success_rate:
                        self.add(other_concept, reproject=False)
                        count += 1
                elif conflict_resolution == 'merge_words':
                    existing.add_words(other_concept.words)
                    existing.text_snippets.extend(other_concept.text_snippets)
                    count += 1
        
        # Reproject after all additions
        if count > 0:
            self._reproject()
        
        return count
    
    def save(self, path: str) -> None:
        """
        Save the store to a JSON file.
        
        Args:
            path: File path to save to
        """
        data = {
            'version': '1.0',
            'type': 'geometric_knowledge_store',
            'metadata': {
                'name': self.name,
                'dims': self.dims,
                'tier': self.tier,
                'concept_count': len(self.concepts),
                'created': self.created,
                'modified': self.modified,
            },
            'geometry': {
                'similarity_matrix': self.similarity_matrix.tolist() if self.similarity_matrix is not None else None,
                'positions': self.positions.tolist() if self.positions is not None else None,
            },
            'concepts': [c.to_dict() for c in self.concepts],
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'GeometricKnowledgeStore':
        """
        Load a store from a JSON file.
        
        Args:
            path: File path to load from
            
        Returns:
            The loaded store
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        geometry = data.get('geometry', {})
        
        store = cls(
            name=metadata.get('name', 'loaded'),
            dims=metadata.get('dims', 12),
            tier=metadata.get('tier', 'temporary'),
            created=metadata.get('created', datetime.now().isoformat()),
            modified=metadata.get('modified', datetime.now().isoformat()),
        )
        
        # Load concepts
        for concept_data in data.get('concepts', []):
            concept = Concept.from_dict(concept_data)
            store.concepts.append(concept)
        
        # Load geometry
        if geometry.get('similarity_matrix'):
            store.similarity_matrix = np.array(geometry['similarity_matrix'])
        if geometry.get('positions'):
            store.positions = np.array(geometry['positions'])
        
        # Rebuild indices
        store._rebuild_indices()
        
        return store
    
    def __len__(self) -> int:
        return len(self.concepts)
    
    def __repr__(self) -> str:
        temp_count = len(self.get_temporary_concepts())
        perm_count = len(self.get_permanent_concepts())
        return f"GeometricKnowledgeStore({self.name}, {perm_count} permanent, {temp_count} temporary)"
    
    def __contains__(self, concept_id: str) -> bool:
        return concept_id in self._id_to_index
