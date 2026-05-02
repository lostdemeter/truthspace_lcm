"""
Navigator: The O(1) Lookup Interface
=====================================

The Navigator brings together:
1. φ-Coordinates (universal representation)
2. Paths (concept-specific transformations)
3. Relationships (abstract definitions)

It provides the unified interface for semantic navigation.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

try:
    from .coordinates import PhiCoordinates, PhiPoint
    from .paths import PathStore, RelationshipPath
    from .relationships import Relationship, RELATIONSHIP_REGISTRY
except ImportError:
    from coordinates import PhiCoordinates, PhiPoint
    from paths import PathStore, RelationshipPath
    from relationships import Relationship, RELATIONSHIP_REGISTRY


@dataclass
class NavigationResult:
    """Result of a navigation query."""
    source: str
    relationship: str
    target: str
    method: str  # 'path_lookup', 'navigation', 'generation'
    confidence: float
    alternatives: List[Tuple[str, float]] = None
    
    def __str__(self):
        return f"{self.source} --[{self.relationship}]--> {self.target} ({self.method}, {self.confidence:.2f})"


class PhiNavigator:
    """
    Main navigator class.
    
    Provides three methods for answering queries:
    1. PATH LOOKUP (O(1), 100% accurate if path exists)
    2. NAVIGATION (O(dim), uses φ-transformation)
    3. GENERATION (O(tokens), uses model inference)
    
    Falls back through these methods in order.
    """
    
    def __init__(self, model=None, tokenizer=None, path_store: Optional[PathStore] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device if model else None
        
        self.coordinates = PhiCoordinates()
        self.path_store = path_store or PathStore()
        
        # Cache for embeddings
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        self._phi_cache: Dict[str, PhiPoint] = {}
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        """Get embedding for a word, with caching."""
        if word in self._embedding_cache:
            return self._embedding_cache[word]
        
        if self.model is None:
            return None
        
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        
        embed = self.model.model.embed_tokens.weight[ids[0]].detach()
        self._embedding_cache[word] = embed
        return embed
    
    def get_phi_point(self, word: str) -> Optional[PhiPoint]:
        """Get φ-coordinates for a word, with caching."""
        if word in self._phi_cache:
            return self._phi_cache[word]
        
        embed = self.get_embedding(word)
        if embed is None:
            return None
        
        point = self.coordinates.encode(embed.cpu())
        self._phi_cache[word] = point
        return point
    
    def find_nearest(self, embed: torch.Tensor, top_k: int = 5,
                     exclude: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Find nearest tokens to an embedding."""
        if self.model is None:
            return []
        
        all_embeds = self.model.model.embed_tokens.weight.detach()
        sims = F.cosine_similarity(embed.unsqueeze(0).float().to(self.device),
                                   all_embeds.float())
        
        if exclude:
            for word in exclude:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if ids:
                    sims[ids[0]] = -1
        
        top_indices = sims.topk(top_k).indices
        return [(self.tokenizer.decode([idx.item()]).strip(), sims[idx].item())
                for idx in top_indices]
    
    # =========================================================================
    # METHOD 1: PATH LOOKUP (O(1), 100% accurate)
    # =========================================================================
    
    def lookup(self, source: str, relationship: str) -> Optional[NavigationResult]:
        """
        O(1) path lookup.
        
        Returns the stored target if a path exists, None otherwise.
        """
        target = self.path_store.get_target(source, relationship)
        if target:
            return NavigationResult(
                source=source,
                relationship=relationship,
                target=target,
                method='path_lookup',
                confidence=1.0,
            )
        return None
    
    # =========================================================================
    # METHOD 2: NAVIGATION (O(dim), uses φ-transformation)
    # =========================================================================
    
    def navigate(self, source: str, relationship: str,
                 path: Optional[RelationshipPath] = None) -> Optional[NavigationResult]:
        """
        Navigate using φ-transformation.
        
        If a path is provided, applies that transformation.
        Otherwise, tries to find a similar path to use.
        """
        source_point = self.get_phi_point(source)
        if source_point is None:
            return None
        
        if path:
            # Apply the specific path
            target_point = path.apply(source_point)
        else:
            # Try to find a path for this relationship
            stored_path = self.path_store.get(source, relationship)
            if stored_path:
                target_point = stored_path.apply(source_point)
            else:
                return None
        
        # Decode and find nearest
        target_embed = target_point.to_embedding().to(self.device)
        nearest = self.find_nearest(target_embed, top_k=5, exclude=[source])
        
        if nearest:
            return NavigationResult(
                source=source,
                relationship=relationship,
                target=nearest[0][0],
                method='navigation',
                confidence=nearest[0][1],
                alternatives=nearest[1:],
            )
        return None
    
    # =========================================================================
    # METHOD 3: GENERATION (O(tokens), uses model inference)
    # =========================================================================
    
    def generate(self, source: str, relationship: Relationship) -> Optional[NavigationResult]:
        """
        Generate answer using model inference.
        
        This is the fallback when no path exists.
        Also used to discover new paths.
        """
        if self.model is None:
            return None
        
        prompt = relationship.get_validation_prompt(source)
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        # Extract first word
        target = response.split()[0].strip(".,!?\"'") if response.split() else ""
        
        if target:
            return NavigationResult(
                source=source,
                relationship=relationship.name,
                target=target.lower(),
                method='generation',
                confidence=0.9,  # Model confidence assumed high
            )
        return None
    
    # =========================================================================
    # UNIFIED INTERFACE
    # =========================================================================
    
    def query(self, source: str, relationship: str,
              relationship_obj: Optional[Relationship] = None) -> NavigationResult:
        """
        Unified query interface.
        
        Tries methods in order:
        1. Path lookup (O(1), 100% accurate)
        2. Navigation (O(dim), approximate)
        3. Generation (O(tokens), uses model)
        
        Returns the first successful result.
        """
        # Method 1: Path lookup
        result = self.lookup(source, relationship)
        if result:
            return result
        
        # Method 2: Navigation (if we have a stored path)
        result = self.navigate(source, relationship)
        if result:
            return result
        
        # Method 3: Generation (fallback)
        if relationship_obj:
            result = self.generate(source, relationship_obj)
            if result:
                return result
        
        # Nothing worked
        return NavigationResult(
            source=source,
            relationship=relationship,
            target="[unknown]",
            method='failed',
            confidence=0.0,
        )
    
    # =========================================================================
    # PATH DISCOVERY
    # =========================================================================
    
    def discover_path(self, source: str, target: str, 
                      relationship: str) -> RelationshipPath:
        """
        Discover the path from source to target.
        
        Computes the exact transformation in φ-space.
        """
        source_point = self.get_phi_point(source)
        target_point = self.get_phi_point(target)
        
        if source_point is None or target_point is None:
            raise ValueError(f"Cannot get embeddings for {source} or {target}")
        
        level_delta, flip_mask = self.coordinates.diff(source_point, target_point)
        flip_dims = flip_mask.nonzero().squeeze().tolist()
        if isinstance(flip_dims, int):
            flip_dims = [flip_dims]
        
        return RelationshipPath(
            source=source,
            target=target,
            relationship=relationship,
            level_delta=level_delta.tolist(),
            flip_dims=flip_dims,
            validated=True,
            confidence=1.0,
        )
    
    def discover_relationship(self, relationship: Relationship,
                              n_pairs: int = 15) -> List[RelationshipPath]:
        """
        Discover paths for a relationship type.
        
        Uses the model to generate example pairs, then computes paths.
        """
        if self.model is None:
            raise ValueError("Model required for discovery")
        
        # Get example pairs from model
        prompt = relationship.get_discovery_prompt(n_pairs)
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        # Parse pairs
        import re
        pairs = []
        for line in response.strip().split('\n'):
            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    w1 = re.sub(r'[^a-zA-Z]', '', parts[0]).strip().lower()
                    w2 = re.sub(r'[^a-zA-Z]', '', parts[1]).strip().lower()
                    if w1 and w2 and w1 != w2 and len(w1) > 1 and len(w2) > 1:
                        pairs.append((w1, w2))
        
        # Discover paths for each pair
        paths = []
        for source, target in pairs[:n_pairs]:
            try:
                path = self.discover_path(source, target, relationship.name)
                paths.append(path)
                self.path_store.add(path, symmetric=relationship.symmetric)
            except Exception as e:
                continue
        
        return paths
