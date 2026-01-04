"""
Concept - The Atomic Unit of Geometric Knowledge

A Concept represents a single unit of knowledge in the geometric space.
It has:
- A position (quaternion) in concept space
- Surface forms (words that represent it)
- Metadata (usage statistics, hierarchy, source)

The key insight: Position IS identity. Two concepts at the same position
are the same concept, regardless of their surface forms.

Author: Lesley Gushurst
License: GPLv3
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import uuid


class ConceptLevel(Enum):
    """Hierarchical level of a concept."""
    FACT = "fact"           # Atomic fact (e.g., "Washington was first president")
    CLUSTER = "cluster"     # Group of related facts (e.g., "Founding Fathers")
    TOPIC = "topic"         # High-level topic (e.g., "American Revolution")


@dataclass
class Concept:
    """
    The atomic unit of geometric knowledge.
    
    A concept is defined by its position in quaternion space, not by its text.
    The words are just surface forms - different words can map to the same concept,
    and the same word can map to different concepts (polysemy).
    
    Attributes:
        id: Unique identifier
        words: Set of content words associated with this concept
        quaternion: Position in 4D concept space (w, x, y, z)
        position_index: Index in the store's position matrix (set by store)
        
        level: Hierarchical level (fact, cluster, topic)
        parent_id: ID of parent concept (for hierarchy)
        
        use_count: How many times this concept has been accessed
        success_count: How many times access led to successful outcome
        stability: How stable the position has been (0.0-1.0)
        
        created: When the concept was created
        modified: When the concept was last modified
        source: Where this concept came from
        
        temporary: Whether this is a temporary (session) concept
    """
    
    # Core identity
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    words: Set[str] = field(default_factory=set)
    
    # Geometric position (quaternion: w, x, y, z)
    quaternion: tuple = field(default=(1.0, 0.0, 0.0, 0.0))
    position_index: int = -1  # Set by store when added
    
    # Hierarchy
    level: ConceptLevel = ConceptLevel.FACT
    parent_id: Optional[str] = None
    
    # Usage statistics
    use_count: int = 0
    success_count: int = 0
    stability: float = 1.0
    
    # Metadata
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "unknown"
    
    # Tier
    temporary: bool = True
    
    # Optional text cache (for debugging/display)
    text_snippets: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate from usage statistics."""
        if self.use_count == 0:
            return 0.5  # Default for unused concepts
        return self.success_count / self.use_count
    
    @property
    def qualifies_for_promotion(self) -> bool:
        """
        Check if this concept qualifies for promotion to permanent.
        
        Criteria (from Design 088):
        - use_count >= 5
        - success_rate >= 0.8
        - stability >= 0.9
        """
        return (
            self.use_count >= 5 and
            self.success_rate >= 0.8 and
            self.stability >= 0.9
        )
    
    def record_use(self, success: bool = True) -> None:
        """Record a use of this concept."""
        self.use_count += 1
        if success:
            self.success_count += 1
        self.modified = datetime.now().isoformat()
    
    def update_position(self, new_quaternion: tuple, drift_threshold: float = 0.1) -> None:
        """
        Update the concept's position and track stability.
        
        Stability decreases if the position drifts significantly.
        """
        if self.quaternion != (1.0, 0.0, 0.0, 0.0):  # Not default
            # Calculate drift (Euclidean distance in quaternion space)
            drift = sum((a - b) ** 2 for a, b in zip(self.quaternion, new_quaternion)) ** 0.5
            
            # Update stability based on drift
            if drift > drift_threshold:
                self.stability = max(0.0, self.stability - 0.1)
            else:
                self.stability = min(1.0, self.stability + 0.01)
        
        self.quaternion = new_quaternion
        self.modified = datetime.now().isoformat()
    
    def add_words(self, new_words: Set[str]) -> None:
        """Add words to this concept's surface forms."""
        self.words.update(new_words)
        self.modified = datetime.now().isoformat()
    
    def promote(self) -> None:
        """Promote this concept from temporary to permanent."""
        self.temporary = False
        self.modified = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'words': list(self.words),
            'quaternion': list(self.quaternion),
            'position_index': self.position_index,
            'level': self.level.value,
            'parent_id': self.parent_id,
            'use_count': self.use_count,
            'success_count': self.success_count,
            'stability': self.stability,
            'created': self.created,
            'modified': self.modified,
            'source': self.source,
            'temporary': self.temporary,
            'text_snippets': self.text_snippets,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Concept':
        """Create a Concept from a dictionary."""
        return cls(
            id=data.get('id', str(uuid.uuid4())[:8]),
            words=set(data.get('words', [])),
            quaternion=tuple(data.get('quaternion', [1.0, 0.0, 0.0, 0.0])),
            position_index=data.get('position_index', -1),
            level=ConceptLevel(data.get('level', 'fact')),
            parent_id=data.get('parent_id'),
            use_count=data.get('use_count', 0),
            success_count=data.get('success_count', 0),
            stability=data.get('stability', 1.0),
            created=data.get('created', datetime.now().isoformat()),
            modified=data.get('modified', datetime.now().isoformat()),
            source=data.get('source', 'unknown'),
            temporary=data.get('temporary', True),
            text_snippets=data.get('text_snippets', []),
        )
    
    def __repr__(self) -> str:
        words_preview = ', '.join(list(self.words)[:3])
        if len(self.words) > 3:
            words_preview += '...'
        tier = "temp" if self.temporary else "perm"
        return f"Concept({self.id}, [{words_preview}], {tier}, uses={self.use_count})"
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Concept):
            return False
        return self.id == other.id
