# Design Consideration 100: φ-Lattice Implementation Plan

## Date: 2026-01-06

## Status: Implementation Plan

## Overview

This document provides a detailed implementation plan for integrating φ-lattice coordinates into the current `truthspace_lcm` workspace, as proposed in Design 099.

## Current Architecture

### Key Components

```
truthspace_lcm/
├── core/
│   ├── knowledge_space.py      # KnowledgeSpace (extends HyperMapping)
│   ├── chat_pipeline.py        # ChatPipeline, IntentSpace
│   ├── bootstrap_knowledge.py  # Loads bootstrap JSON
│   ├── code_space.py          # Code generation
│   ├── plot_space.py          # Plot generation
│   └── ollama_space.py        # LLM integration
├── corpus/
│   └── bootstrap_knowledge.json
└── practical_applications/
    └── chat/
        └── hyper_api.py       # API server

hypermapping/
├── hypermapping.py            # HyperMapping, Mapping, Encoder base
└── encoders.py                # TextEncoder, NumericEncoder, etc.
```

### Current Flow

1. **TextEncoder** encodes text to positions via word hashing + averaging
2. **KnowledgeSpace** stores Mappings with eigenspace-derived positions
3. **query_text()** projects query to eigenspace, finds nearest by distance
4. **Sqrt-inverse weighting** applied to distance calculation (88% accuracy)

### The Problem

- Positions are **relative** (derived from similarity matrix eigendecomposition)
- **DC component** captures 58% of variance
- Positions compressed to narrow range [0.1, 0.5]
- No semantic meaning in dimensions

## Proposed Architecture

### New Components

```
truthspace_lcm/
├── core/
│   ├── phi_lattice.py          # NEW: φ-lattice coordinate system
│   ├── semantic_dimensions.py  # NEW: Semantic dimension definitions
│   ├── primitives.py           # NEW: Primitive definitions (from old TruthSpace)
│   ├── knowledge_space.py      # MODIFY: Add φ-lattice mode
│   ├── chat_pipeline.py        # MODIFY: Use φ-lattice for intent
│   └── bootstrap_knowledge.py  # MODIFY: Add lattice positions
├── corpus/
│   └── bootstrap_knowledge.json # MODIFY: Add lattice level assignments
```

## Implementation Phases

### Phase 1: Core φ-Lattice Infrastructure

**File: `truthspace_lcm/core/phi_lattice.py`**

```python
"""
φ-Lattice Coordinate System

Provides absolute coordinates based on φ^k levels with semantic dimensions.
Replaces relative eigenspace coordinates for knowledge matching.

Design Principles:
- Positions at φ^k for integer k (absolute, verifiable)
- Semantic dimensions with clear meaning
- No DC component - positions aren't derived from similarity
- Full dynamic range: φ^(-10) to φ^(+10)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618


@dataclass
class SemanticDimension:
    """Definition of a semantic dimension."""
    index: int
    name: str
    description: str
    level_meanings: Dict[int, str]  # level -> meaning


class PhiLattice:
    """
    φ-Lattice coordinate system with semantic dimensions.
    
    Positions are at φ^k for integer k on each dimension.
    Each dimension has semantic meaning defined at initialization.
    """
    
    def __init__(self, dimensions: List[SemanticDimension]):
        self.dimensions = {d.index: d for d in dimensions}
        self.ndim = len(dimensions)
        
        # Precompute φ-levels for efficiency
        self._phi_levels = {k: PHI ** k for k in range(-10, 11)}
    
    def levels_to_position(self, levels: List[int]) -> np.ndarray:
        """Convert φ-level indices to position vector."""
        return np.array([PHI ** k for k in levels])
    
    def position_to_levels(self, position: np.ndarray) -> List[int]:
        """Convert position to nearest φ-level indices."""
        levels = []
        for v in position:
            best_k = round(np.log(abs(v) + 1e-10) / np.log(PHI))
            best_k = max(-10, min(10, best_k))
            levels.append(best_k)
        return levels
    
    def snap_to_lattice(self, position: np.ndarray) -> np.ndarray:
        """Snap position to nearest valid φ-lattice point."""
        levels = self.position_to_levels(position)
        return self.levels_to_position(levels)
    
    def is_valid_position(self, position: np.ndarray, 
                          tolerance: float = 0.01) -> bool:
        """Check if position is on the φ-lattice."""
        snapped = self.snap_to_lattice(position)
        return np.allclose(position, snapped, atol=tolerance)
    
    def distance(self, a: np.ndarray, b: np.ndarray, 
                 weights: Optional[np.ndarray] = None) -> float:
        """
        Weighted Euclidean distance in φ-space.
        
        Default weights follow old TruthSpace pattern:
        - Higher weight for action dimensions
        - Lower weight for relation dimensions
        """
        if weights is None:
            weights = np.ones(len(a))
        diff = (a - b) * weights
        return float(np.linalg.norm(diff))
```

### Phase 2: Semantic Dimension Definitions

**File: `truthspace_lcm/core/semantic_dimensions.py`**

```python
"""
Semantic Dimension Definitions for Knowledge Matching

Defines the semantic axes of the φ-lattice:
- DOMAIN: What area of knowledge
- SPECIFICITY: How specific is the concept
- INTENT: What kind of response expected
- FORMALITY: How formal is the context

Each dimension has φ-levels with semantic meanings.
"""

from .phi_lattice import SemanticDimension

# Domain dimension: What area of knowledge
DOMAIN = SemanticDimension(
    index=0,
    name='domain',
    description='What area of knowledge',
    level_meanings={
        3: 'hard_science',      # Physics, Math, Chemistry
        2: 'technology',        # Programming, Engineering
        1: 'general_knowledge', # General facts
        0: 'meta',              # Identity, self-reference
        -1: 'social',           # Greetings, thanks
    }
)

# Specificity dimension: How specific is the concept
SPECIFICITY = SemanticDimension(
    index=1,
    name='specificity',
    description='How specific is the concept',
    level_meanings={
        3: 'very_specific',     # Quantum mechanics, specific algorithm
        2: 'specific',          # Physics, Python
        1: 'general',           # Science, programming
        0: 'very_general',      # Knowledge, information
        -1: 'vague',            # Anything, something
    }
)

# Intent dimension: What kind of response expected
INTENT = SemanticDimension(
    index=2,
    name='intent',
    description='What kind of response is expected',
    level_meanings={
        2: 'explanation',       # Teach, explain in detail
        1: 'information',       # Facts, data
        0: 'acknowledgment',    # Confirm, acknowledge
        -1: 'social_response',  # Greeting response, thanks response
    }
)

# Formality dimension: How formal is the context
FORMALITY = SemanticDimension(
    index=3,
    name='formality',
    description='How formal is the context',
    level_meanings={
        2: 'academic',          # Technical, formal
        1: 'professional',      # Business, clear
        0: 'casual',            # Everyday, relaxed
        -1: 'informal',         # Friendly, chatty
    }
)

# Default dimension set for knowledge matching
DEFAULT_DIMENSIONS = [DOMAIN, SPECIFICITY, INTENT, FORMALITY]

# Dimension weights (inspired by old TruthSpace PHI_BLOCK_WEIGHTS)
# Higher weight for domain/specificity, lower for formality
DEFAULT_WEIGHTS = {
    0: PHI ** 2,   # Domain: most important
    1: PHI,        # Specificity: important
    2: 1.0,        # Intent: neutral
    3: PHI ** -1,  # Formality: least important
}
```

### Phase 3: Primitive Definitions

**File: `truthspace_lcm/core/primitives.py`**

```python
"""
Primitives for φ-Lattice Encoding

Primitives are semantic anchors that map words to φ-lattice positions.
Inspired by the old TruthSpace implementation.

Each primitive has:
- name: Identifier
- dimension: Which semantic dimension it activates
- level: What φ-level it activates
- keywords: Words that trigger this primitive
"""

from dataclasses import dataclass
from typing import List, Dict

PHI = 1.618033988749895


@dataclass
class Primitive:
    """A semantic anchor in the φ-lattice."""
    name: str
    dimension: int
    level: int
    keywords: List[str]


# Domain primitives (dimension 0)
DOMAIN_PRIMITIVES = [
    Primitive("PHYSICS", 0, 3, ["physics", "quantum", "relativity", "mechanics"]),
    Primitive("MATH", 0, 3, ["math", "mathematics", "calculus", "algebra"]),
    Primitive("CHEMISTRY", 0, 3, ["chemistry", "chemical", "molecule"]),
    Primitive("PROGRAMMING", 0, 2, ["programming", "code", "python", "software"]),
    Primitive("TECHNOLOGY", 0, 2, ["technology", "computer", "digital"]),
    Primitive("GENERAL", 0, 1, ["knowledge", "information", "learn"]),
    Primitive("IDENTITY", 0, 0, ["you", "your", "yourself", "hyperchat"]),
    Primitive("SOCIAL", 0, -1, ["hello", "hi", "thanks", "thank", "goodbye", "bye"]),
]

# Specificity primitives (dimension 1)
SPECIFICITY_PRIMITIVES = [
    Primitive("VERY_SPECIFIC", 1, 3, ["quantum", "differential", "neural"]),
    Primitive("SPECIFIC", 1, 2, ["physics", "python", "machine"]),
    Primitive("GENERAL", 1, 1, ["science", "programming", "learning"]),
    Primitive("VERY_GENERAL", 1, 0, ["what", "how", "why", "explain"]),
    Primitive("VAGUE", 1, -1, ["something", "anything", "stuff"]),
]

# Intent primitives (dimension 2)
INTENT_PRIMITIVES = [
    Primitive("EXPLAIN", 2, 2, ["explain", "describe", "teach", "how"]),
    Primitive("INFORM", 2, 1, ["what", "tell", "show", "is"]),
    Primitive("ACKNOWLEDGE", 2, 0, ["ok", "yes", "sure", "got"]),
    Primitive("SOCIAL", 2, -1, ["hello", "hi", "thanks", "bye"]),
]

# Formality primitives (dimension 3)
FORMALITY_PRIMITIVES = [
    Primitive("ACADEMIC", 3, 2, ["theory", "hypothesis", "analysis"]),
    Primitive("PROFESSIONAL", 3, 1, ["please", "could", "would"]),
    Primitive("CASUAL", 3, 0, ["can", "want", "need"]),
    Primitive("INFORMAL", 3, -1, ["hey", "yo", "cool"]),
]

# All primitives
ALL_PRIMITIVES = (
    DOMAIN_PRIMITIVES + 
    SPECIFICITY_PRIMITIVES + 
    INTENT_PRIMITIVES + 
    FORMALITY_PRIMITIVES
)


def build_keyword_map() -> Dict[str, Primitive]:
    """Build keyword → primitive mapping."""
    keyword_map = {}
    for prim in ALL_PRIMITIVES:
        for kw in prim.keywords:
            # If multiple primitives have same keyword, prefer higher level
            if kw not in keyword_map or prim.level > keyword_map[kw].level:
                keyword_map[kw.lower()] = prim
    return keyword_map
```

### Phase 4: φ-Lattice Encoder

**File: `truthspace_lcm/core/phi_encoder.py`**

```python
"""
φ-Lattice Encoder

Encodes text to φ-lattice positions using primitives.
Replaces TextEncoder for knowledge matching.
"""

import re
import numpy as np
from typing import List, Set, Dict, Optional

from .phi_lattice import PhiLattice, PHI
from .semantic_dimensions import DEFAULT_DIMENSIONS, DEFAULT_WEIGHTS
from .primitives import build_keyword_map, Primitive


class PhiLatticeEncoder:
    """
    Encodes text to φ-lattice positions.
    
    Uses primitive detection to determine which semantic dimensions
    are activated and at what level.
    
    Encoding follows MAX aggregation (Sierpinski property):
    - Multiple words activating same dimension → take max level
    - Position decay by word order (later words contribute less)
    """
    
    def __init__(self, lattice: Optional[PhiLattice] = None):
        if lattice is None:
            lattice = PhiLattice(DEFAULT_DIMENSIONS)
        self.lattice = lattice
        self.keyword_map = build_keyword_map()
        self.weights = np.array([DEFAULT_WEIGHTS.get(i, 1.0) 
                                 for i in range(lattice.ndim)])
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to words."""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to φ-lattice position.
        
        Uses MAX aggregation per dimension with position decay.
        """
        words = self.tokenize(text)
        levels = [0] * self.lattice.ndim  # Default: neutral position
        
        for i, word in enumerate(words):
            if word in self.keyword_map:
                prim = self.keyword_map[word]
                dim = prim.dimension
                level = prim.level
                
                # Position decay: later words contribute less
                # But still use MAX aggregation
                decay_factor = PHI ** (-i * 0.5)
                effective_level = level  # Level doesn't decay, just influence
                
                # MAX aggregation (Sierpinski property)
                if effective_level > levels[dim]:
                    levels[dim] = effective_level
        
        return self.lattice.levels_to_position(levels)
    
    def encode_with_levels(self, text: str) -> tuple:
        """Encode and return both position and levels."""
        words = self.tokenize(text)
        levels = [0] * self.lattice.ndim
        
        for i, word in enumerate(words):
            if word in self.keyword_map:
                prim = self.keyword_map[word]
                if prim.level > levels[prim.dimension]:
                    levels[prim.dimension] = prim.level
        
        position = self.lattice.levels_to_position(levels)
        return position, levels
    
    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Weighted distance in φ-space."""
        return self.lattice.distance(a, b, self.weights)
```

### Phase 5: Modify KnowledgeSpace

**Modify: `truthspace_lcm/core/knowledge_space.py`**

Add a `use_phi_lattice` mode that uses the new encoder:

```python
# Add to imports
from .phi_lattice import PhiLattice
from .phi_encoder import PhiLatticeEncoder
from .semantic_dimensions import DEFAULT_DIMENSIONS

class KnowledgeSpace(HyperMapping):
    def __init__(self, name: str = "knowledge", dims: int = 8,
                 use_phi_lattice: bool = False):
        # ... existing init ...
        
        self.use_phi_lattice = use_phi_lattice
        if use_phi_lattice:
            self._phi_lattice = PhiLattice(DEFAULT_DIMENSIONS)
            self._phi_encoder = PhiLatticeEncoder(self._phi_lattice)
            # Override dims to match lattice
            self._dims = self._phi_lattice.ndim
    
    def query_text(self, text: str, top_k: int = 5,
                   min_similarity: float = 0.0) -> List[MatchResult]:
        if self.use_phi_lattice:
            return self._query_phi_lattice(text, top_k, min_similarity)
        else:
            return self._query_eigenspace(text, top_k, min_similarity)
    
    def _query_phi_lattice(self, text: str, top_k: int,
                           min_similarity: float) -> List[MatchResult]:
        """Query using φ-lattice coordinates."""
        query_pos = self._phi_encoder.encode(text)
        
        results = []
        for mapping in self._mappings:
            # Get concept's φ-lattice position
            if hasattr(mapping, 'phi_position'):
                concept_pos = mapping.phi_position
            else:
                # Encode concept text to φ-lattice
                concept_pos = self._phi_encoder.encode(mapping.input)
            
            distance = self._phi_encoder.distance(query_pos, concept_pos)
            similarity = 1.0 / (1.0 + distance)
            
            results.append(MatchResult(
                mapping=mapping,
                similarity=float(similarity),
            ))
        
        results.sort(key=lambda r: -r.similarity)
        
        if min_similarity > 0:
            results = [r for r in results if r.similarity >= min_similarity]
        
        return results[:top_k]
```

### Phase 6: Update Bootstrap Knowledge

**Modify: `truthspace_lcm/corpus/bootstrap_knowledge.json`**

Add φ-lattice level assignments to each knowledge item:

```json
{
  "knowledge": [
    {
      "text": "Physics is the natural science that studies matter...",
      "topic": "physics",
      "phi_levels": [3, 2, 2, 1],
      "keywords": ["physics", "matter", "energy"]
    },
    {
      "text": "What can I do for you? I am HyperChat...",
      "topic": "identity",
      "phi_levels": [0, 0, 1, 0],
      "keywords": ["you", "hyperchat", "help"]
    },
    {
      "text": "Hello! I'm HyperChat. How can I help you?",
      "topic": "greeting",
      "phi_levels": [-1, -1, -1, -1],
      "keywords": ["hello", "hi", "help"]
    }
  ]
}
```

### Phase 7: Integration and Testing

**File: `truthspace_lcm/core/tests/test_phi_lattice.py`**

```python
"""Tests for φ-lattice implementation."""

import numpy as np
from ..phi_lattice import PhiLattice, PHI
from ..phi_encoder import PhiLatticeEncoder
from ..semantic_dimensions import DEFAULT_DIMENSIONS


def test_phi_lattice_basic():
    """Test basic φ-lattice operations."""
    lattice = PhiLattice(DEFAULT_DIMENSIONS)
    
    # Test levels to position
    levels = [2, 1, 0, -1]
    pos = lattice.levels_to_position(levels)
    expected = np.array([PHI**2, PHI**1, PHI**0, PHI**-1])
    assert np.allclose(pos, expected)
    
    # Test position to levels
    recovered = lattice.position_to_levels(pos)
    assert recovered == levels
    
    # Test validity
    assert lattice.is_valid_position(pos)
    
    noisy = pos + np.random.randn(4) * 0.1
    assert not lattice.is_valid_position(noisy)


def test_phi_encoder():
    """Test φ-lattice encoder."""
    encoder = PhiLatticeEncoder()
    
    # Physics query should activate domain=3, specificity=2
    pos, levels = encoder.encode_with_levels("what is physics?")
    assert levels[0] >= 2  # Domain: physics
    assert levels[2] >= 1  # Intent: information (what)
    
    # Identity query should activate domain=0
    pos, levels = encoder.encode_with_levels("who are you?")
    assert levels[0] == 0  # Domain: identity
    
    # Greeting should activate domain=-1
    pos, levels = encoder.encode_with_levels("hello")
    assert levels[0] == -1  # Domain: social


def test_knowledge_matching():
    """Test knowledge matching with φ-lattice."""
    from ..knowledge_space import KnowledgeSpace
    
    space = KnowledgeSpace(use_phi_lattice=True)
    
    # Add test knowledge
    space.add_text("Physics is the study of matter and energy", 
                   source="test", phi_levels=[3, 2, 2, 1])
    space.add_text("Hello! How can I help you?",
                   source="test", phi_levels=[-1, -1, -1, -1])
    
    # Test queries
    results = space.query_text("what is physics?")
    assert "physics" in results[0].mapping.input.lower()
    
    results = space.query_text("hello")
    assert "hello" in results[0].mapping.input.lower()
```

## Migration Strategy

### Option A: Parallel Mode (Recommended)

Run both eigenspace and φ-lattice in parallel:

```python
class KnowledgeSpace:
    def __init__(self, ..., mode: str = "eigenspace"):
        # mode: "eigenspace", "phi_lattice", or "hybrid"
        self.mode = mode
```

This allows:
1. A/B testing between approaches
2. Gradual migration
3. Fallback if φ-lattice fails

### Option B: Feature Flag

Use a feature flag to switch modes:

```python
# In chat_pipeline.py
USE_PHI_LATTICE = os.environ.get("USE_PHI_LATTICE", "false").lower() == "true"
```

### Option C: Hybrid Navigation

Use similarity for direction, φ-lattice for position:

```python
def query_hybrid(self, text):
    # Get direction from similarity
    similarities = self._compute_similarities(text)
    
    # Get position from φ-lattice
    phi_pos = self._phi_encoder.encode(text)
    
    # Combine: weight lattice distance by similarity
    scores = {}
    for mapping in self._mappings:
        lattice_dist = self._phi_encoder.distance(phi_pos, mapping.phi_position)
        similarity = similarities[mapping.id]
        scores[mapping.id] = similarity / (1 + lattice_dist)
    
    return max(scores, key=scores.get)
```

## File Summary

### New Files to Create

| File | Purpose |
|------|---------|
| `core/phi_lattice.py` | φ-lattice coordinate system |
| `core/semantic_dimensions.py` | Semantic dimension definitions |
| `core/primitives.py` | Primitive definitions |
| `core/phi_encoder.py` | Text to φ-lattice encoder |
| `core/tests/test_phi_lattice.py` | Unit tests |

### Files to Modify

| File | Changes |
|------|---------|
| `core/knowledge_space.py` | Add `use_phi_lattice` mode |
| `core/chat_pipeline.py` | Use φ-lattice for intent detection |
| `core/bootstrap_knowledge.py` | Load φ-levels from JSON |
| `corpus/bootstrap_knowledge.json` | Add `phi_levels` to each item |

## Success Criteria

1. **No DC Component**: Positions are absolute, not similarity-derived
2. **Full Dynamic Range**: φ^(-10) to φ^(+10) instead of [0.1, 0.5]
3. **Verifiable Positions**: Can check if position is valid lattice point
4. **Semantic Dimensions**: Each axis has clear meaning
5. **Accuracy**: Match or exceed 88% on test queries
6. **No Regression**: Existing functionality continues to work

## Timeline

| Phase | Description | Estimate |
|-------|-------------|----------|
| 1 | Core φ-lattice infrastructure | 1 hour |
| 2 | Semantic dimensions | 30 min |
| 3 | Primitives | 30 min |
| 4 | φ-lattice encoder | 1 hour |
| 5 | KnowledgeSpace integration | 1 hour |
| 6 | Bootstrap knowledge update | 30 min |
| 7 | Testing and validation | 1 hour |
| **Total** | | **~5.5 hours** |

## Next Steps

1. Create `core/phi_lattice.py` with basic infrastructure
2. Create `core/semantic_dimensions.py` with dimension definitions
3. Create `core/primitives.py` with primitive definitions
4. Create `core/phi_encoder.py` with encoder
5. Modify `core/knowledge_space.py` to support φ-lattice mode
6. Update `corpus/bootstrap_knowledge.json` with φ-levels
7. Test and validate

---

*"The φ-lattice provides the coordinate system. Similarity provides the navigation. Structure IS information."*
