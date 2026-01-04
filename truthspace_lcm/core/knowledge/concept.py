"""
Concept - The Atomic Unit of Geometric Knowledge

A Concept represents a single unit of knowledge in the geometric space.

The key insight (Design 091): POSITION IS EVERYTHING.

A concept has:
- A position (quaternion) in concept space - this IS the concept
- Surface forms (words) - just labels for the position
- Created/modified timestamps - for debugging only

Everything else is DERIVED from position:
- Magnitude = confidence (distance from origin)
- Persistence = magnitude >= 0.5 (critical line)
- Stability = low variance in position over time (tracked by store)

The critical line (σ = 0.5) is the information horizon.
Concepts past the horizon persist. Concepts inside fade.

Author: Lesley Gushurst
License: GPLv3
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Any
from datetime import datetime
import numpy as np
import uuid


# Critical line constant - the information horizon
CRITICAL_LINE = 0.5


@dataclass
class Concept:
    """
    The atomic unit of geometric knowledge.
    
    POSITION IS EVERYTHING (Design 091).
    
    A concept is defined entirely by its position. The words are just
    surface forms - labels that point to the position.
    
    Everything is derived from position:
    - magnitude: How "strong" the concept is
    - persists: Whether it's past the critical line (σ = 0.5)
    - At origin: New/unused concept
    - Past critical line: Established concept
    
    Attributes:
        id: Unique identifier
        words: Set of content words (surface forms)
        position: N-dimensional position vector (the concept's identity)
        created: When created (for debugging)
        modified: When last modified (for debugging)
    """
    
    # Core identity
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    words: Set[str] = field(default_factory=set)
    
    # THE position - this IS the concept
    # Stored as tuple for immutability, converted to numpy for math
    position: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0))
    
    # Metadata (for debugging/display only)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "unknown"
    text_snippets: List[str] = field(default_factory=list)
    
    @property
    def position_array(self) -> np.ndarray:
        """Position as numpy array for math operations."""
        return np.array(self.position)
    
    @property
    def magnitude(self) -> float:
        """
        Magnitude of position vector.
        
        This IS the concept's "strength" or "confidence".
        - 0.0 = at origin (new/unused)
        - 0.5 = at critical line (threshold)
        - 1.0+ = well-established
        """
        return float(np.linalg.norm(self.position_array))
    
    @property
    def persists(self) -> bool:
        """
        Whether this concept persists (is past the critical line).
        
        The critical line (σ = 0.5) is the information horizon.
        Concepts past it have enough "weight" to persist.
        Concepts inside it will fade.
        """
        return self.magnitude >= CRITICAL_LINE
    
    @property
    def normalized_position(self) -> tuple:
        """
        Position normalized to unit sphere.
        
        Useful for direction-only comparisons.
        """
        mag = self.magnitude
        if mag < 1e-10:
            return self.position
        arr = self.position_array / mag
        return tuple(arr)
    
    def move_toward(self, target: tuple, strength: float = 0.1) -> None:
        """
        Move this concept toward a target position.
        
        This is the ONLY learning operation needed.
        Called on successful use - pulls concept toward query position.
        
        Args:
            target: Target position to move toward
            strength: How much to move (0.0-1.0)
        """
        current = self.position_array
        target_arr = np.array(target)
        
        # Move toward target
        new_pos = current + strength * (target_arr - current)
        
        self.position = tuple(new_pos)
        self.modified = datetime.now().isoformat()
    
    def move_away(self, target: tuple, strength: float = 0.05) -> None:
        """
        Move this concept away from a target position.
        
        Called on failed use - pushes concept away from query position.
        
        Args:
            target: Target position to move away from
            strength: How much to move (0.0-1.0)
        """
        current = self.position_array
        target_arr = np.array(target)
        
        # Move away from target
        new_pos = current - strength * (target_arr - current)
        
        self.position = tuple(new_pos)
        self.modified = datetime.now().isoformat()
    
    def add_words(self, new_words: Set[str]) -> None:
        """Add words to this concept's surface forms."""
        self.words.update(new_words)
        self.modified = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'words': list(self.words),
            'position': list(self.position),
            'created': self.created,
            'modified': self.modified,
            'source': self.source,
            'text_snippets': self.text_snippets,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Concept':
        """Create a Concept from a dictionary."""
        # Handle legacy 'quaternion' field
        position = data.get('position') or data.get('quaternion', [0.0, 0.0, 0.0, 0.0])
        return cls(
            id=data.get('id', str(uuid.uuid4())[:8]),
            words=set(data.get('words', [])),
            position=tuple(position),
            created=data.get('created', datetime.now().isoformat()),
            modified=data.get('modified', datetime.now().isoformat()),
            source=data.get('source', 'unknown'),
            text_snippets=data.get('text_snippets', []),
        )
    
    def __repr__(self) -> str:
        words_preview = ', '.join(list(self.words)[:3])
        if len(self.words) > 3:
            words_preview += '...'
        status = "persists" if self.persists else "fading"
        return f"Concept({self.id}, [{words_preview}], mag={self.magnitude:.2f}, {status})"
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Concept):
            return False
        return self.id == other.id
