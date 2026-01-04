"""GeometricKnowledgeStore - Persistent Geometric Knowledge Storage

The core insight (Design 091): POSITION IS EVERYTHING.

This store manages:
1. Concepts (position + words)
2. Similarity matrix (word overlap between concepts)
3. Positions (derived from similarity via eigendecomposition)

Key Operations:
- add(): Add a concept at the origin
- use(): THE learning operation - move concept based on success/failure
- query(): Find nearest concepts by word overlap or position
- prune(): Remove concepts inside the critical line
- save()/load(): Persist to/from JSON

Design Principles (from Design 091):
- POSITION IS IDENTITY
- MOVEMENT IS LEARNING
- THE CRITICAL LINE IS THE HORIZON

Concepts start at the origin. Success moves them toward query positions.
Failure moves them away. Concepts past the critical line (σ = 0.5) persist.
Concepts inside the critical line fade and can be pruned.

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
from collections import Counter

from .concept import Concept, CRITICAL_LINE


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
    dims: int = 4  # Default to 4D (quaternion-like)
    
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
    
    # Internal flag for deferred reprojection
    _needs_reproject: bool = field(default=False, repr=False)
    
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
    
    def extract_words(self, text: str) -> Set[str]:
        """
        Extract content words from text using geometric stop word detection.
        
        Geometric principle (from Design 062):
        - Content words have clear semantic roles (appear in specific contexts)
        - Stop words have NO semantic role (appear everywhere uniformly)
        - Detection is based on distribution, not a hardcoded list
        
        Returns lowercase content words.
        """
        # Tokenize: split on non-alphanumeric
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter using geometric detection
        content_words = {
            w for w in words 
            if self._is_content_word(w)
        }
        
        return content_words
    
    def _is_content_word(self, word: str) -> bool:
        """
        Geometric stop word detection.
        
        A word is a STOP word (not content) if:
        1. It's very short (< 3 chars) - structural, not semantic
        2. It appears in many concepts but adds no discriminative power
        
        This is geometric because it's based on the word's distribution
        across the concept space, not a hardcoded list.
        """
        # Very short words are structural, not semantic
        if len(word) < 3:
            return False
        
        # If we have concept data, use it for geometric detection
        if word in self._word_to_concepts and len(self.concepts) > 1:
            # Word appears in what fraction of concepts?
            coverage = len(self._word_to_concepts[word]) / len(self.concepts)
            
            # High coverage (> 50%) = stop word (appears everywhere)
            # This threshold is the critical line (σ = 0.5)
            if coverage > 0.5:
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
            
            # Check if full reproject is needed (dimension mismatch was detected)
            if self._needs_reproject:
                self._reproject()
                self._needs_reproject = False
        
        self.modified = datetime.now().isoformat()
    
    def add_from_text(self, text: str, source: str = "text") -> Concept:
        """
        Create and add a concept from text.
        
        The concept starts at the origin. It will move based on
        successful/failed uses via the use() method.
        
        Args:
            text: The text to create a concept from
            source: Source attribution
            
        Returns:
            The created concept
        """
        words = self.extract_words(text)
        
        concept = Concept(
            words=words,
            source=source,
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
            # Use concept's existing position if set, otherwise origin
            if new_concept.magnitude > 0:
                pos = np.array(new_concept.position)
                if len(pos) != self.dims:
                    pos = np.pad(pos, (0, max(0, self.dims - len(pos))))[:self.dims]
                self.positions = pos.reshape(1, -1)
            else:
                self.positions = np.zeros((1, self.dims))
                new_concept.position = tuple(np.zeros(self.dims))
            return
        
        # Check if positions matrix needs reinitialization (dimension mismatch)
        if self.positions is None or self.positions.shape[1] != self.dims:
            # Reinitialize positions with correct dimensions
            self.positions = np.zeros((n-1, self.dims))
            # Trigger full reproject after this add
            self._needs_reproject = True
        
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
        
        # Use concept's existing position if set, otherwise use computed position
        if new_concept.magnitude > 0:
            # Concept already has a position - use it
            pos = np.array(new_concept.position)
            if len(pos) != self.dims:
                pos = np.pad(pos, (0, max(0, self.dims - len(pos))))[:self.dims]
            new_position = pos
        
        # Extend positions matrix
        self.positions = np.vstack([self.positions, new_position])
        
        # Update concept's position
        new_concept.position = tuple(new_position)
    
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
        
        # Update concept positions from the projected space
        for i, concept in enumerate(self.concepts):
            concept.position = tuple(self.positions[i])
    
    def use(self, concept_id: str, query_position: tuple, success: bool,
            attract_strength: float = 0.1, repel_strength: float = 0.05) -> bool:
        """
        THE learning operation. Move concept based on success/failure.
        
        This is the ONLY way concepts learn. Everything else emerges:
        - Frequently successful concepts move toward query clusters
        - Failed concepts drift away
        - Concepts past the critical line persist
        - Concepts inside the critical line fade
        
        Args:
            concept_id: ID of the concept that was used
            query_position: Position of the query that matched this concept
            success: Whether the use was successful
            attract_strength: How much to move toward on success (default 0.1)
            repel_strength: How much to move away on failure (default 0.05)
            
        Returns:
            True if concept was found and updated, False otherwise
        """
        concept = self.get(concept_id)
        if concept is None:
            return False
        
        if success:
            concept.move_toward(query_position, attract_strength)
        else:
            concept.move_away(query_position, repel_strength)
        
        self.modified = datetime.now().isoformat()
        return True
    
    def get_persisting_concepts(self) -> List[Concept]:
        """Get all concepts past the critical line (will persist)."""
        return [c for c in self.concepts if c.persists]
    
    def get_fading_concepts(self) -> List[Concept]:
        """Get all concepts inside the critical line (will fade)."""
        return [c for c in self.concepts if not c.persists]
    
    def prune(self, threshold: float = None) -> int:
        """
        Remove concepts inside the critical line.
        
        This is the natural garbage collection - concepts that haven't
        moved past the critical line are pruned.
        
        Args:
            threshold: Magnitude threshold (default: CRITICAL_LINE = 0.5)
            
        Returns:
            Number of concepts removed
        """
        if threshold is None:
            threshold = CRITICAL_LINE
        
        to_remove = [c.id for c in self.concepts if c.magnitude < threshold]
        for cid in to_remove:
            self.remove(cid)
        
        return len(to_remove)
    
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
                elif conflict_resolution == 'higher_magnitude':
                    # Use position magnitude (concepts further from origin are stronger)
                    if other_concept.magnitude > existing.magnitude:
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
                'concept_count': len(self.concepts),
                'persisting_count': len(self.get_persisting_concepts()),
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
            dims=metadata.get('dims', 4),
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
