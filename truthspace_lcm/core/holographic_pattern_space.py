"""
Holographic Pattern Space

A pattern matching system that constructs geometry from similarity.
Instead of encoding text and hoping similar things land close,
we define similarity explicitly and construct positions that realize it.

Key principle: The similarity matrix IS the structure.
Positions are just a representation of that structure in vector space.

Author: Lesley Gushurst
License: GPLv3
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HolographicModule:
    """A module with holographically projected position."""
    name: str
    text: str
    words: Set[str]
    module_type: str  # 'enhancer', 'promoter', 'temporary'
    effects: Dict[str, Any] = field(default_factory=dict)
    target_gears: Set[str] = field(default_factory=set)
    code_template: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    use_count: int = 0
    success_count: int = 0
    position: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'text': self.text,
            'words': list(self.words),
            'module_type': self.module_type,
            'effects': self.effects,
            'target_gears': list(self.target_gears),
            'code_template': self.code_template,
            'examples': self.examples,
            'use_count': self.use_count,
            'success_count': self.success_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HolographicModule':
        return cls(
            name=data['name'],
            text=data['text'],
            words=set(data.get('words', [])),
            module_type=data.get('module_type', 'enhancer'),
            effects=data.get('effects', {}),
            target_gears=set(data.get('target_gears', [])),
            code_template=data.get('code_template'),
            examples=data.get('examples', []),
            use_count=data.get('use_count', 0),
            success_count=data.get('success_count', 0),
        )


class HolographicPatternSpace:
    """
    Pattern space where positions are CONSTRUCTED from similarity.
    
    Key principle: The similarity matrix IS the structure.
    Positions are just a representation of that structure in vector space.
    
    Given modules M with similarity S[i,j] = word_overlap(M[i], M[j]):
    - Eigendecompose S to get positions P
    - Now: dot(P[i], P[j]) ≈ S[i,j] by construction!
    
    Features:
    - Geometry encodes similarity directly (no gates needed)
    - Queries project based on their similarity to known modules
    - Zero overlap → zero projection → correctly rejected
    - Temporary module injection for unknown queries
    - Learning via promotion of successful temporaries
    """
    
    FILLER = {'a', 'an', 'the', 'of', 'with', 'for', 'to', 'and', 'or', 'in', 
              'create', 'make', 'generate', 'plot', 'that', 'this', 'is', 'are'}
    
    def __init__(self, dims: int = 12):
        self.dims = dims
        self.modules: List[HolographicModule] = []
        self.positions: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[np.ndarray] = None
    
    @classmethod
    def extract_words(cls, text: str) -> Set[str]:
        """Extract content words from text."""
        words = text.lower().split()
        return {w for w in words if w not in cls.FILLER and len(w) > 1}
    
    @classmethod
    def word_overlap(cls, words1: Set[str], words2: Set[str]) -> float:
        """
        Calculate word overlap (asymmetric coverage).
        
        Returns: fraction of words1 that are in words2
        """
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        return len(intersection) / len(words1)
    
    @classmethod
    def symmetric_overlap(cls, words1: Set[str], words2: Set[str]) -> float:
        """Symmetric Jaccard similarity for module-to-module."""
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
    
    def add_module(self, name: str, text: str, module_type: str, 
                   effects: Dict[str, Any] = None,
                   target_gears: Set[str] = None,
                   code_template: str = None,
                   examples: List[str] = None) -> HolographicModule:
        """Add a module and reproject all positions."""
        words = self.extract_words(text)
        module = HolographicModule(
            name=name,
            text=text,
            words=words,
            module_type=module_type,
            effects=effects or {},
            target_gears=target_gears or set(),
            code_template=code_template,
            examples=examples or [],
        )
        self.modules.append(module)
        self._reproject()
        return module
    
    def _reproject(self):
        """
        Construct positions from similarity matrix using eigendecomposition.
        
        This is the key: we DEFINE what similarity means (word overlap),
        then CONSTRUCT positions that realize that similarity.
        """
        n = len(self.modules)
        if n == 0:
            self.positions = None
            self.similarity_matrix = None
            return
        
        # Build similarity matrix
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    S[i, j] = 1.0
                else:
                    S[i, j] = self.symmetric_overlap(
                        self.modules[i].words, 
                        self.modules[j].words
                    )
        
        self.similarity_matrix = S
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take top dims dimensions
        k = min(self.dims, n)
        eigenvalues_k = np.maximum(eigenvalues[:k], 0)  # Clamp negatives
        
        self.positions = eigenvectors[:, :k] * np.sqrt(eigenvalues_k)
        
        # Pad to full dims if needed
        if k < self.dims:
            padding = np.zeros((n, self.dims - k))
            self.positions = np.hstack([self.positions, padding])
        
        # Store positions in modules
        for i, module in enumerate(self.modules):
            module.position = self.positions[i]
    
    def project_query(self, query_text: str) -> np.ndarray:
        """
        Project a query into the space.
        
        The query's position is determined by its similarity to each module.
        """
        query_words = self.extract_words(query_text)
        n = len(self.modules)
        
        if n == 0 or self.positions is None:
            return np.zeros(self.dims)
        
        # Compute similarity of query to each module
        similarities = np.array([
            self.word_overlap(query_words, m.words) 
            for m in self.modules
        ])
        
        # Weighted average of module positions
        if np.sum(similarities) > 0:
            query_pos = similarities @ self.positions / (np.sum(similarities) + 1e-10)
        else:
            query_pos = np.zeros(self.dims)
        
        return query_pos
    
    def find_nearest(self, query_text: str, k: int = 3) -> List[Tuple[HolographicModule, float, float]]:
        """
        Find nearest modules to query.
        
        Returns: List of (module, distance, similarity)
        """
        query_pos = self.project_query(query_text)
        query_words = self.extract_words(query_text)
        
        results = []
        for i, module in enumerate(self.modules):
            if module.position is not None:
                dist = np.linalg.norm(query_pos - module.position)
            else:
                dist = float('inf')
            sim = self.word_overlap(query_words, module.words)
            results.append((module, dist, sim))
        
        results.sort(key=lambda x: x[1])
        return results[:k]
    
    def find_best_match(self, query_text: str, 
                        min_similarity: float = 0.3) -> Tuple[Optional[HolographicModule], float, str]:
        """
        Find best match with confidence assessment.
        
        Returns: (module, confidence, reason)
        """
        query_words = self.extract_words(query_text)
        
        if not query_words:
            return None, 0.0, "empty query"
        
        results = self.find_nearest(query_text, k=5)
        
        if not results:
            return None, 0.0, "no modules"
        
        best_module, best_dist, best_sim = results[0]
        
        # Check if there's any word overlap
        if best_sim == 0:
            return None, 0.0, "no word overlap"
        
        # Check minimum similarity
        if best_sim < min_similarity:
            return best_module, best_sim, "weak match"
        
        # Confidence based on similarity and separation
        if len(results) >= 2:
            second_sim = results[1][2]
            if second_sim > 0:
                ratio = best_sim / second_sim
                if ratio < 1.5:
                    return best_module, best_sim * 0.7, f"ambiguous (ratio={ratio:.2f})"
        
        return best_module, best_sim, "strong match"
    
    def inject_temporary_module(self, query_text: str, 
                                 fallback_effects: Dict[str, Any] = None) -> HolographicModule:
        """
        Inject a temporary module based on the query itself.
        
        This handles the "George Washington" problem:
        - Query contains words not in any existing module
        - We create a temporary module from the query
        - Reproject the space to include it
        - The query will now match this temporary module
        """
        query_words = self.extract_words(query_text)
        
        # Generate unique name
        import hashlib
        query_hash = hashlib.md5(query_text.lower().encode()).hexdigest()[:8]
        name = f"temp_{query_hash}"
        
        temp_module = HolographicModule(
            name=name,
            text=query_text,
            words=query_words,
            module_type='temporary',
            effects=fallback_effects or {'task': 'llm_fallback'},
            examples=[query_text],
        )
        
        self.modules.append(temp_module)
        self._reproject()
        
        return temp_module
    
    def remove_temporary_modules(self):
        """Remove all temporary modules."""
        self.modules = [m for m in self.modules if m.module_type != 'temporary']
        if self.modules:
            self._reproject()
    
    def find_or_inject(self, query_text: str, 
                       fallback_effects: Dict[str, Any] = None,
                       min_similarity: float = 0.3) -> Tuple[HolographicModule, float, str, bool]:
        """
        Find best match, or inject temporary module if no match.
        
        Returns: (module, confidence, reason, was_injected)
        """
        # First, try to find a match
        module, confidence, reason = self.find_best_match(query_text, min_similarity)
        
        if module is not None and confidence >= min_similarity:
            return module, confidence, reason, False
        
        # No good match - check if we should inject
        query_words = self.extract_words(query_text)
        
        if not query_words:
            return None, 0.0, "empty query", False
        
        # Check max overlap with any module
        max_overlap = max(
            (self.word_overlap(query_words, m.words) for m in self.modules),
            default=0.0
        )
        
        if max_overlap == 0:
            # No overlap at all - inject temporary module
            temp_module = self.inject_temporary_module(query_text, fallback_effects)
            return temp_module, 1.0, "injected temporary", True
        
        # Some overlap but below threshold
        if module:
            return module, confidence, f"weak match ({reason})", False
        
        return None, 0.0, "no suitable match", False
    
    def promote_temporary(self, module: HolographicModule, 
                          new_type: str = 'enhancer',
                          new_effects: Dict[str, Any] = None,
                          code_template: str = None):
        """
        Promote a temporary module to permanent.
        Called when LLM successfully handled the query.
        """
        if module.module_type == 'temporary':
            module.module_type = new_type
            if new_effects:
                module.effects.update(new_effects)
            if code_template:
                module.code_template = code_template
            module.success_count += 1
    
    def record_use(self, module: HolographicModule, success: bool):
        """Record module usage."""
        module.use_count += 1
        if success:
            module.success_count += 1
    
    def get_module_by_name(self, name: str) -> Optional[HolographicModule]:
        """Get a module by name."""
        for m in self.modules:
            if m.name == name:
                return m
        return None
    
    def save(self, path: str):
        """Save space to JSON file."""
        data = {
            'dims': self.dims,
            'modules': [m.to_dict() for m in self.modules if m.module_type != 'temporary']
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load space from JSON file."""
        if not Path(path).exists():
            return
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.dims = data.get('dims', 12)
        self.modules = [HolographicModule.from_dict(m) for m in data.get('modules', [])]
        
        if self.modules:
            self._reproject()
