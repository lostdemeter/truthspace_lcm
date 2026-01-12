#!/usr/bin/env python3
"""
Modality-Agnostic Core

This module provides the truly universal foundation that works for ANY modality:
- Text
- Images
- Audio
- Video
- 3D models
- Code
- Any other transformable data

The key insight: The φ-geometry doesn't care about modality.
A dimension is just a transformation between two states.
The SAME self-assembly mechanism works for all modalities.

Author: TruthSpace LCM Project
License: GPLv3
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, TypeVar, Generic, Callable
from enum import Enum, auto
import hashlib

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# MODALITY ENUM
# =============================================================================

class Modality(Enum):
    """
    Supported modalities.
    
    Each modality has its own scale hierarchy and dimension types,
    but they all use the same φ-geometry.
    """
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    CODE = auto()
    MESH_3D = auto()
    GENERIC = auto()  # For unknown/custom modalities


# =============================================================================
# ABSTRACT ARTIFACT
# =============================================================================

# Type variable for artifact content
T = TypeVar('T')


@dataclass
class Artifact(Generic[T]):
    """
    A modality-agnostic artifact.
    
    An artifact is any piece of content that can be transformed:
    - Text: a string
    - Image: a numpy array (H, W, C)
    - Audio: a numpy array (samples,)
    - etc.
    
    The artifact carries its content plus metadata about its modality.
    """
    content: T
    modality: Modality
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def content_hash(self) -> str:
        """Hash of the content for identity."""
        if isinstance(self.content, np.ndarray):
            return hashlib.md5(self.content.tobytes()).hexdigest()[:16]
        elif isinstance(self.content, str):
            return hashlib.md5(self.content.encode()).hexdigest()[:16]
        else:
            return hashlib.md5(str(self.content).encode()).hexdigest()[:16]
    
    @property
    def identifier(self) -> str:
        """Unique identifier for this artifact."""
        return f"{self.modality.name}:{self.content_hash}"


# =============================================================================
# ABSTRACT TRANSFORM
# =============================================================================

@dataclass
class Transform(Generic[T]):
    """
    A modality-agnostic transformation.
    
    A transform takes an artifact and produces a new artifact.
    This is the fundamental unit of a dimension.
    
    Examples:
    - Text: uppercase, formal→casual, king→queen
    - Image: grayscale, blur, rotate, invert
    - Audio: pitch_shift, reverb, normalize
    """
    name: str
    modality: Modality
    transform_fn: Callable[[T], T]
    inverse_fn: Optional[Callable[[T], T]] = None
    
    # Metadata
    description: str = ""
    is_reversible: bool = False
    
    def apply(self, artifact: Artifact[T]) -> Artifact[T]:
        """Apply the transform to an artifact."""
        new_content = self.transform_fn(artifact.content)
        return Artifact(
            content=new_content,
            modality=artifact.modality,
            metadata={**artifact.metadata, 'transform': self.name}
        )
    
    def reverse(self, artifact: Artifact[T]) -> Optional[Artifact[T]]:
        """Reverse the transform if possible."""
        if self.inverse_fn is None:
            return None
        new_content = self.inverse_fn(artifact.content)
        return Artifact(
            content=new_content,
            modality=artifact.modality,
            metadata={**artifact.metadata, 'transform': f'inverse_{self.name}'}
        )


# =============================================================================
# ABSTRACT DIMENSION
# =============================================================================

@dataclass
class UniversalDimension(Generic[T]):
    """
    A modality-agnostic dimension.
    
    A dimension is defined by:
    1. A transformation (the "what")
    2. Two poles (the endpoints)
    3. A modality (what kind of content)
    4. A scale (what level of granularity)
    
    The φ-geometry is the same regardless of modality.
    """
    name: str
    modality: Modality
    negative_pole: Artifact[T]
    positive_pole: Artifact[T]
    transform: Optional[Transform[T]] = None
    scale: int = 0  # 0=finest, higher=coarser
    
    @property
    def delta(self) -> float:
        """The standard delta for this dimension (always φ)."""
        return PHI


# =============================================================================
# UNIVERSAL CORPUS
# =============================================================================

@dataclass
class UniversalPair(Generic[T]):
    """A transformation pair in any modality."""
    source: Artifact[T]
    target: Artifact[T]
    dimension: str
    modality: Modality


class UniversalCorpus:
    """
    A modality-agnostic corpus.
    
    This is the truly universal version that works for any modality.
    The φ-geometry is identical regardless of what's being transformed.
    """
    
    def __init__(self):
        # Pairs by modality
        self._pairs: Dict[Modality, List[UniversalPair]] = {m: [] for m in Modality}
        
        # Dimensions by modality
        self._dimensions: Dict[Modality, Dict[str, UniversalDimension]] = {m: {} for m in Modality}
        
        # Positions (artifact_id → position vector)
        self._positions: Dict[str, np.ndarray] = {}
        
        # Dimension order (for consistent indexing)
        self._dimension_order: List[str] = []
        
        # Transforms registry
        self._transforms: Dict[Modality, Dict[str, Transform]] = {m: {} for m in Modality}
    
    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------
    
    def register_transform(self, transform: Transform):
        """Register a transform for a modality."""
        self._transforms[transform.modality][transform.name] = transform
    
    def register_dimension(self, dimension: UniversalDimension):
        """Register a dimension."""
        self._dimensions[dimension.modality][dimension.name] = dimension
        
        if dimension.name not in self._dimension_order:
            self._dimension_order.append(dimension.name)
    
    # -------------------------------------------------------------------------
    # Adding Pairs
    # -------------------------------------------------------------------------
    
    def add_pair(self, source: Artifact, target: Artifact, 
                 dimension: str) -> bool:
        """Add a transformation pair."""
        if source.modality != target.modality:
            return False  # Cross-modality pairs not supported yet
        
        modality = source.modality
        
        pair = UniversalPair(
            source=source,
            target=target,
            dimension=dimension,
            modality=modality
        )
        
        self._pairs[modality].append(pair)
        
        # Ensure dimension exists
        if dimension not in self._dimension_order:
            self._dimension_order.append(dimension)
        
        # Position the artifacts
        self._position_pair(pair)
        
        return True
    
    def _position_pair(self, pair: UniversalPair):
        """Position a pair in the geometry."""
        dim_idx = self._dimension_order.index(pair.dimension)
        n_dims = len(self._dimension_order)
        
        # Source at 0, target at φ along this dimension
        src_id = pair.source.identifier
        tgt_id = pair.target.identifier
        
        # Extend ALL positions to current dimension count
        for artifact_id in list(self._positions.keys()):
            if len(self._positions[artifact_id]) < n_dims:
                old = self._positions[artifact_id]
                self._positions[artifact_id] = np.zeros(n_dims)
                self._positions[artifact_id][:len(old)] = old
        
        if src_id not in self._positions:
            self._positions[src_id] = np.zeros(n_dims)
        if tgt_id not in self._positions:
            self._positions[tgt_id] = np.zeros(n_dims)
        
        # Set positions
        self._positions[tgt_id][dim_idx] = self._positions[src_id][dim_idx] + PHI
    
    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------
    
    def get_position(self, artifact: Artifact) -> Optional[np.ndarray]:
        """Get the position of an artifact."""
        return self._positions.get(artifact.identifier)
    
    def find_nearest(self, position: np.ndarray, modality: Modality = None,
                     n: int = 5) -> List[Tuple[str, float]]:
        """Find nearest artifacts to a position."""
        results = []
        
        for artifact_id, pos in self._positions.items():
            # Filter by modality if specified
            if modality:
                artifact_modality = Modality[artifact_id.split(':')[0]]
                if artifact_modality != modality:
                    continue
            
            # Pad to same length
            max_len = max(len(position), len(pos))
            p1 = np.pad(position, (0, max_len - len(position)))
            p2 = np.pad(pos, (0, max_len - len(pos)))
            
            distance = np.linalg.norm(p1 - p2)
            results.append((artifact_id, distance))
        
        results.sort(key=lambda x: x[1])
        return results[:n]
    
    def get_dimensions(self, modality: Modality = None) -> List[str]:
        """Get dimensions, optionally filtered by modality."""
        if modality is None:
            return self._dimension_order.copy()
        
        return [
            name for name, dim in self._dimensions[modality].items()
        ]
    
    def get_pairs(self, modality: Modality = None) -> List[UniversalPair]:
        """Get pairs, optionally filtered by modality."""
        if modality is None:
            all_pairs = []
            for pairs in self._pairs.values():
                all_pairs.extend(pairs)
            return all_pairs
        
        return self._pairs[modality].copy()
    
    # -------------------------------------------------------------------------
    # Cross-Modality
    # -------------------------------------------------------------------------
    
    def find_analogous_dimension(self, dimension: str, 
                                  from_modality: Modality,
                                  to_modality: Modality) -> Optional[str]:
        """
        Find an analogous dimension in another modality.
        
        Example: "grayscale" in IMAGE might be analogous to "monotone" in AUDIO
        """
        # This is a placeholder - real implementation would use
        # semantic similarity or learned mappings
        
        # For now, check if same dimension name exists
        if dimension in self._dimensions[to_modality]:
            return dimension
        
        return None
    
    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------
    
    def get_status(self) -> Dict:
        """Get corpus status."""
        return {
            "total_dimensions": len(self._dimension_order),
            "total_artifacts": len(self._positions),
            "pairs_by_modality": {
                m.name: len(pairs) for m, pairs in self._pairs.items()
                if pairs
            },
            "dimensions_by_modality": {
                m.name: len(dims) for m, dims in self._dimensions.items()
                if dims
            },
        }


# =============================================================================
# MODALITY ADAPTER (Abstract Base)
# =============================================================================

class ModalityAdapter(ABC, Generic[T]):
    """
    Abstract base class for modality-specific adapters.
    
    Each modality implements this to provide:
    1. Scale hierarchy for that modality
    2. Default transforms/dimensions
    3. Analysis functions
    4. Serialization
    """
    
    @property
    @abstractmethod
    def modality(self) -> Modality:
        """The modality this adapter handles."""
        pass
    
    @property
    @abstractmethod
    def scales(self) -> List[str]:
        """Scale hierarchy for this modality."""
        pass
    
    @abstractmethod
    def create_artifact(self, content: T, **metadata) -> Artifact[T]:
        """Create an artifact from raw content."""
        pass
    
    @abstractmethod
    def get_default_transforms(self) -> List[Transform[T]]:
        """Get default transforms for this modality."""
        pass
    
    @abstractmethod
    def analyze(self, artifact: Artifact[T]) -> Dict[str, float]:
        """Analyze an artifact to extract dimensional coordinates."""
        pass
    
    @abstractmethod
    def detect_scale(self, artifact: Artifact[T]) -> int:
        """Detect the primary scale of an artifact."""
        pass


# =============================================================================
# DEMO
# =============================================================================

def demo_universal_corpus():
    """Demonstrate the modality-agnostic corpus."""
    print("=" * 70)
    print("DEMO: Universal (Modality-Agnostic) Corpus")
    print("=" * 70)
    print()
    print("The φ-geometry doesn't care about modality.")
    print("A dimension is just a transformation between two states.")
    print()
    
    corpus = UniversalCorpus()
    
    # Add text pairs
    print("Adding TEXT pairs...")
    text_pairs = [
        ("king", "queen", "gender"),
        ("small", "large", "size"),
        ("casual", "formal", "register"),
    ]
    
    for src, tgt, dim in text_pairs:
        src_artifact = Artifact(content=src, modality=Modality.TEXT)
        tgt_artifact = Artifact(content=tgt, modality=Modality.TEXT)
        corpus.add_pair(src_artifact, tgt_artifact, dim)
    
    # Add image pairs (simulated with numpy arrays)
    print("Adding IMAGE pairs...")
    
    # Simulate color → grayscale
    color_img = np.random.rand(64, 64, 3)  # RGB image
    gray_img = np.mean(color_img, axis=2, keepdims=True).repeat(3, axis=2)
    
    color_artifact = Artifact(content=color_img, modality=Modality.IMAGE,
                              metadata={'type': 'color'})
    gray_artifact = Artifact(content=gray_img, modality=Modality.IMAGE,
                             metadata={'type': 'grayscale'})
    corpus.add_pair(color_artifact, gray_artifact, "saturation")
    
    # Simulate sharp → blurred
    sharp_img = np.random.rand(64, 64, 3)
    # Simple box blur simulation
    blurred_img = np.zeros_like(sharp_img)
    for i in range(1, 63):
        for j in range(1, 63):
            blurred_img[i, j] = sharp_img[i-1:i+2, j-1:j+2].mean(axis=(0, 1))
    
    sharp_artifact = Artifact(content=sharp_img, modality=Modality.IMAGE,
                              metadata={'type': 'sharp'})
    blur_artifact = Artifact(content=blurred_img, modality=Modality.IMAGE,
                             metadata={'type': 'blurred'})
    corpus.add_pair(sharp_artifact, blur_artifact, "sharpness")
    
    # Status
    print()
    print("Corpus status:")
    status = corpus.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()
    
    # Show positions
    print("Positions (first 3 dimensions):")
    for artifact_id, pos in list(corpus._positions.items())[:6]:
        pos_str = np.array2string(pos[:3], precision=2)
        print(f"  {artifact_id[:30]:30} → {pos_str}")
    print()
    
    # Verify φ-delta
    print("Verifying φ-delta for all pairs:")
    for modality, pairs in corpus._pairs.items():
        for pair in pairs:
            src_pos = corpus.get_position(pair.source)
            tgt_pos = corpus.get_position(pair.target)
            if src_pos is not None and tgt_pos is not None:
                delta = np.linalg.norm(tgt_pos - src_pos)
                match = "✓" if abs(delta - PHI) < 0.01 else "✗"
                print(f"  {pair.dimension}: Δ = {delta:.3f} {match}")
    print()
    
    print("=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print()
    print("The SAME φ-geometry works for:")
    print("  - Text: king → queen (gender dimension)")
    print("  - Image: color → grayscale (saturation dimension)")
    print("  - Image: sharp → blurred (sharpness dimension)")
    print()
    print("The modality doesn't matter. The geometry is universal.")
    print()
    
    return corpus


if __name__ == "__main__":
    demo_universal_corpus()
