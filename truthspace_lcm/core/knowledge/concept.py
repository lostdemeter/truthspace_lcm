"""
Concept - The Atomic Unit of Geometric Knowledge

A Concept represents a single unit of knowledge in the geometric space.
It has:
- A position (quaternion) in concept space
- Surface forms (words that represent it)
- Metadata (usage statistics, hierarchy, source)

The key insight: Position IS identity. Two concepts at the same position
are the same concept, regardless of their surface forms.

Geometric Principles:
- Promotion is based on geometric confidence (neighborhood fit), not hardcoded thresholds
- Stability measures how well the concept fits its attractor basin
- All thresholds emerge from the data's own distribution

Author: Lesley Gushurst
License: GPLv3
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import math
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
    
    # Geometric stability: measures fit to attractor basin
    # Computed as 1 / (1 + average_drift) where drift is position change over time
    cumulative_drift: float = 0.0
    drift_samples: int = 0
    
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
            return 0.5  # Default for unused concepts (critical line)
        return self.success_count / self.use_count
    
    @property
    def stability(self) -> float:
        """
        Geometric stability: how well this concept fits its attractor basin.
        
        Computed as 1 / (1 + average_drift), which naturally:
        - Approaches 1.0 for stable concepts (low drift)
        - Approaches 0.0 for unstable concepts (high drift)
        - Starts at 1.0 for new concepts (no drift yet)
        
        This is geometric because it emerges from the concept's own trajectory.
        """
        if self.drift_samples == 0:
            return 1.0  # New concept, assume stable
        average_drift = self.cumulative_drift / self.drift_samples
        return 1.0 / (1.0 + average_drift)
    
    @property
    def confidence(self) -> float:
        """
        Geometric confidence: combines success rate and stability.
        
        Uses geometric mean (sqrt of product) which:
        - Requires BOTH factors to be high
        - Is scale-invariant (geometric property)
        - Naturally balances the two measures
        """
        return math.sqrt(self.success_rate * self.stability)
    
    def qualifies_for_promotion(self, threshold: float = 0.5) -> bool:
        """
        Check if this concept qualifies for promotion to permanent.
        
        Geometric criteria:
        - Must have been used (use_count > 0)
        - Confidence must meet or exceed the threshold
        
        The threshold should be computed by the store based on the
        population distribution (e.g., critical line at 0.5, or
        a percentile of the confidence distribution).
        
        Args:
            threshold: The confidence threshold for promotion.
                      Defaults to 0.5 (critical line).
        """
        return (
            self.use_count > 0 and
            self.confidence >= threshold
        )
    
    def record_use(self, success: bool = True) -> None:
        """Record a use of this concept."""
        self.use_count += 1
        if success:
            self.success_count += 1
        self.modified = datetime.now().isoformat()
    
    def update_position(self, new_quaternion: tuple) -> None:
        """
        Update the concept's position and track drift.
        
        Drift is accumulated geometrically - we track the actual distance
        moved in quaternion space. Stability emerges from this naturally:
        concepts that move a lot have low stability.
        
        No hardcoded thresholds - stability is computed from cumulative drift.
        """
        if self.quaternion != (1.0, 0.0, 0.0, 0.0):  # Not default position
            # Calculate drift (Euclidean distance in quaternion space)
            drift = sum((a - b) ** 2 for a, b in zip(self.quaternion, new_quaternion)) ** 0.5
            
            # Accumulate drift geometrically
            self.cumulative_drift += drift
            self.drift_samples += 1
        
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
            'cumulative_drift': self.cumulative_drift,
            'drift_samples': self.drift_samples,
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
            cumulative_drift=data.get('cumulative_drift', 0.0),
            drift_samples=data.get('drift_samples', 0),
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
