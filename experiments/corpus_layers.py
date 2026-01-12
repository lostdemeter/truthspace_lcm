#!/usr/bin/env python3
"""
Corpus Layers: Attachable Knowledge System

This module implements the layered corpus architecture described in
design doc 116. It enables:

1. BASE LAYER - Language fundamentals (permanent)
2. DOMAIN LAYERS - Topic-specific knowledge (attachable/detachable)
3. CONTEXT LAYER - Conversation state (ephemeral)

Key principle: Shared dimensions are the hooks between layers.
The dimensions ARE the zinc fingers that bind layers together.

Music Box Principle:
- Base corpus = the mechanism (fixed)
- Domain corpus = interchangeable cylinder (attachable)
- Context = current position (ephemeral)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from pathlib import Path
import numpy as np

from experiments.self_assembling_corpus import (
    SelfAssemblingCorpus,
    TransformationPair,
    Dimension,
    PlatonicIdeal,
    ConceptType,
    PHI,
    SPECIFICITY_IDEAL,
    SPECIFICITY_CATEGORY,
    SPECIFICITY_INSTANCE,
)


@dataclass
class RetrievalResult:
    """Result from a geometric retrieval."""
    concept: str
    distance: float
    layer_name: str
    position: Optional[np.ndarray] = None
    specificity: float = SPECIFICITY_CATEGORY
    
    def __repr__(self):
        return f"RetrievalResult({self.concept}, dist={self.distance:.2f}, layer={self.layer_name})"


class CorpusLayer:
    """
    A layer in the corpus stack.
    
    Each layer wraps a SelfAssemblingCorpus and provides:
    - Dimension mapping to align with base layer
    - Priority for query resolution
    - Attach/detach semantics
    """
    
    def __init__(self, name: str, corpus: SelfAssemblingCorpus = None, 
                 priority: int = 50):
        """
        Create a corpus layer.
        
        Args:
            name: Layer identifier
            corpus: The underlying corpus (created if not provided)
            priority: Query priority (higher = checked first)
        """
        self.name = name
        self.corpus = corpus or SelfAssemblingCorpus()
        self.priority = priority
        
        # Dimension mapping: local_name → base_index
        # This allows domain dimensions to align with base dimensions
        self.dimension_map: Dict[str, int] = {}
        
        # Parent layer (for dimension inheritance)
        self._parent: Optional['CorpusLayer'] = None
        self._attached = False
    
    def attach_to(self, parent: 'CorpusLayer'):
        """
        Attach this layer to a parent layer.
        
        Aligns dimensions: if this layer has a dimension with the same name
        as the parent, they share the same index (hook).
        """
        self._parent = parent
        self._attached = True
        
        # Align dimensions
        self._align_dimensions()
    
    def detach(self):
        """Detach from parent layer."""
        self._parent = None
        self._attached = False
        self.dimension_map.clear()
    
    def _align_dimensions(self):
        """Align this layer's dimensions to parent's dimensions."""
        if not self._parent:
            return
        
        parent_dims = self._parent.corpus.dimensions
        
        for dim_name, dim in self.corpus.dimensions.items():
            if dim_name in parent_dims:
                # Shared dimension - use parent's index
                self.dimension_map[dim_name] = parent_dims[dim_name].index
            else:
                # New dimension - keep local index but offset by parent's count
                # This ensures no index collision
                offset = len(parent_dims)
                self.dimension_map[dim_name] = offset + dim.index
    
    def get_aligned_position(self, concept: str) -> Optional[np.ndarray]:
        """
        Get concept position aligned to parent's dimension space.
        
        If attached, positions are mapped to parent's coordinate system.
        """
        local_pos = self.corpus.get_position(concept)
        if local_pos is None:
            return None
        
        if not self._attached or not self._parent:
            return local_pos
        
        # Create aligned position in parent's dimension space
        parent_dim_count = len(self._parent.corpus.dimensions)
        total_dims = parent_dim_count + len(self.corpus.dimensions)
        aligned = np.zeros(total_dims)
        
        for dim_name, dim in self.corpus.dimensions.items():
            if dim_name in self.dimension_map:
                target_idx = self.dimension_map[dim_name]
                if target_idx < len(aligned) and dim.index < len(local_pos):
                    aligned[target_idx] = local_pos[dim.index]
        
        return aligned
    
    def query(self, position: np.ndarray, n: int = 5) -> List[Tuple[str, float]]:
        """
        Find concepts near a position.
        
        Args:
            position: Query position (in aligned coordinate space)
            n: Number of results
            
        Returns:
            List of (concept, distance) tuples
        """
        # Adjust position to match this layer's dimension count
        layer_dims = len(self.corpus.dimensions)
        if layer_dims == 0:
            return []
        
        if len(position) > layer_dims:
            # Truncate to this layer's dimensions
            adjusted_pos = position[:layer_dims]
        elif len(position) < layer_dims:
            # Pad with zeros
            adjusted_pos = np.pad(position, (0, layer_dims - len(position)))
        else:
            adjusted_pos = position
        
        return self.corpus.find_nearest(adjusted_pos, n=n)
    
    def add_pair(self, source: str, target: str, relationship: str, 
                 confidence: float = 1.0) -> bool:
        """Add a transformation pair to this layer."""
        return self.corpus.add_pair(source, target, relationship, confidence)
    
    def get_stats(self) -> Dict:
        """Get layer statistics."""
        self.corpus.recompute()
        return {
            "name": self.name,
            "priority": self.priority,
            "attached": self._attached,
            "pairs": len(self.corpus.pairs),
            "dimensions": len(self.corpus.dimensions),
            "concepts": len(self.corpus.concepts),
            "ideals": len(self.corpus.ideals),
        }


class ContextLayer(CorpusLayer):
    """
    Ephemeral layer for current conversation context.
    
    This layer:
    - Has highest priority (checked first)
    - Tracks recently mentioned concepts
    - Decays over time (recency weighting)
    - Clears between conversations
    """
    
    def __init__(self, parent: CorpusLayer = None):
        super().__init__("context", priority=100)
        
        # Recency tracking: concept → (mention_count, last_turn)
        self._mentions: Dict[str, Tuple[int, int]] = {}
        self._current_turn = 0
        self._decay_rate = 0.9
        
        if parent:
            self.attach_to(parent)
    
    def add_mention(self, concept: str, position: np.ndarray = None):
        """
        Track a concept mentioned in conversation.
        
        Args:
            concept: The concept mentioned
            position: Optional position (if known)
        """
        if concept in self._mentions:
            count, _ = self._mentions[concept]
            self._mentions[concept] = (count + 1, self._current_turn)
        else:
            self._mentions[concept] = (1, self._current_turn)
    
    def advance_turn(self):
        """Advance conversation turn and decay old mentions."""
        self._current_turn += 1
        self._decay()
    
    def _decay(self):
        """Decay old mentions based on recency."""
        to_remove = []
        for concept, (count, turn) in self._mentions.items():
            age = self._current_turn - turn
            decayed_count = count * (self._decay_rate ** age)
            if decayed_count < 0.1:
                to_remove.append(concept)
            else:
                self._mentions[concept] = (decayed_count, turn)
        
        for concept in to_remove:
            del self._mentions[concept]
    
    def get_context_boost(self, concept: str) -> float:
        """
        Get context boost for a concept.
        
        Returns a multiplier based on how recently/frequently mentioned.
        """
        if concept not in self._mentions:
            return 1.0
        
        count, turn = self._mentions[concept]
        recency = self._decay_rate ** (self._current_turn - turn)
        return 1.0 + (count * recency * 0.5)  # Up to 1.5x boost
    
    def clear(self):
        """Clear context (new conversation)."""
        self._mentions.clear()
        self._current_turn = 0
        self.corpus = SelfAssemblingCorpus()
    
    def get_active_concepts(self) -> List[Tuple[str, float]]:
        """Get currently active concepts with their weights."""
        result = []
        for concept, (count, turn) in self._mentions.items():
            recency = self._decay_rate ** (self._current_turn - turn)
            weight = count * recency
            result.append((concept, weight))
        return sorted(result, key=lambda x: -x[1])


class CorpusStack:
    """
    Stack of corpus layers with unified query interface.
    
    The stack manages:
    - One base layer (permanent, language fundamentals)
    - Multiple domain layers (attachable/detachable)
    - One context layer (ephemeral, conversation state)
    
    Queries traverse all attached layers, respecting priority.
    """
    
    def __init__(self, base: CorpusLayer = None):
        """
        Create a corpus stack.
        
        Args:
            base: The base layer (created if not provided)
        """
        self.base = base or CorpusLayer("base", priority=0)
        self.domains: List[CorpusLayer] = []
        self.context = ContextLayer(self.base)
    
    def attach_domain(self, domain: CorpusLayer):
        """
        Attach a domain layer to the stack.
        
        The domain's dimensions are aligned to the base layer.
        """
        domain.attach_to(self.base)
        self.domains.append(domain)
        self.domains.sort(key=lambda d: -d.priority)  # Higher priority first
    
    def detach_domain(self, name: str) -> Optional[CorpusLayer]:
        """
        Detach a domain layer by name.
        
        Returns the detached layer (or None if not found).
        """
        for i, domain in enumerate(self.domains):
            if domain.name == name:
                domain.detach()
                return self.domains.pop(i)
        return None
    
    def get_domain(self, name: str) -> Optional[CorpusLayer]:
        """Get a domain layer by name."""
        for domain in self.domains:
            if domain.name == name:
                return domain
        return None
    
    def all_layers(self) -> List[CorpusLayer]:
        """Get all layers in priority order."""
        layers = [self.context] + self.domains + [self.base]
        return sorted(layers, key=lambda l: -l.priority)
    
    def query(self, text: str, n: int = 10) -> List[RetrievalResult]:
        """
        Query all layers for concepts related to text.
        
        Args:
            text: Query text
            n: Max results per layer
            
        Returns:
            List of RetrievalResult sorted by priority then distance
        """
        # Parse concepts from text
        concepts = self._parse_concepts(text)
        
        if not concepts:
            return []
        
        # Compute query position from known concepts
        position = self._compute_position(concepts)
        
        if position is None:
            return []
        
        # Query all layers
        results = []
        for layer in self.all_layers():
            nearby = layer.query(position, n=n)
            for concept, distance in nearby:
                # Apply context boost
                boost = self.context.get_context_boost(concept)
                adjusted_distance = distance / boost
                
                results.append(RetrievalResult(
                    concept=concept,
                    distance=adjusted_distance,
                    layer_name=layer.name,
                    position=layer.corpus.get_position(concept)
                ))
        
        # Sort by priority (implicit in layer order) then distance
        # Group by layer priority, then sort within group by distance
        results.sort(key=lambda r: r.distance)
        
        # Deduplicate (same concept from multiple layers)
        seen = set()
        unique = []
        for r in results:
            if r.concept not in seen:
                seen.add(r.concept)
                unique.append(r)
        
        return unique[:n]
    
    def _parse_concepts(self, text: str) -> List[str]:
        """Extract known concepts from text."""
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        
        known = []
        for word in words:
            # Check all layers for this concept
            for layer in self.all_layers():
                if layer.corpus.get_position(word) is not None:
                    known.append(word)
                    break
        
        return known
    
    def _compute_position(self, concepts: List[str]) -> Optional[np.ndarray]:
        """Compute geometric position from concepts."""
        if not concepts:
            return None
        
        positions = []
        for concept in concepts:
            for layer in self.all_layers():
                pos = layer.corpus.get_position(concept)
                if pos is not None:
                    positions.append(pos)
                    break
        
        if not positions:
            return None
        
        # Pad to same length
        max_len = max(len(p) for p in positions)
        padded = [np.pad(p, (0, max_len - len(p))) for p in positions]
        
        return np.mean(padded, axis=0)
    
    def traverse(self, start: str, dimension: str, 
                 direction: float = PHI) -> Optional[str]:
        """
        Traverse from a concept along a dimension.
        
        This is what traditional RAG CAN'T do:
        - "What's the opposite of X?"
        - "What's more Y than X?"
        
        Args:
            start: Starting concept
            dimension: Dimension to traverse
            direction: How far to move (default: φ)
            
        Returns:
            The concept at the target position (or None)
        """
        # Find start position
        start_pos = None
        start_layer = None
        for layer in self.all_layers():
            pos = layer.corpus.get_position(start)
            if pos is not None:
                start_pos = pos
                start_layer = layer
                break
        
        if start_pos is None:
            return None
        
        # Find dimension index
        dim_idx = None
        for layer in self.all_layers():
            dim = layer.corpus.get_dimension(dimension)
            if dim:
                dim_idx = dim.index
                break
        
        if dim_idx is None:
            return None
        
        # Compute target position
        target_pos = start_pos.copy()
        if dim_idx < len(target_pos):
            target_pos[dim_idx] += direction
        
        # Find nearest concept to target
        results = self.query_position(target_pos, n=1, exclude=[start])
        if results:
            return results[0].concept
        
        return None
    
    def query_position(self, position: np.ndarray, n: int = 5,
                       exclude: List[str] = None) -> List[RetrievalResult]:
        """Query all layers for concepts near a position."""
        exclude = exclude or []
        results = []
        
        for layer in self.all_layers():
            nearby = layer.query(position, n=n + len(exclude))
            for concept, distance in nearby:
                if concept not in exclude:
                    results.append(RetrievalResult(
                        concept=concept,
                        distance=distance,
                        layer_name=layer.name,
                        position=layer.corpus.get_position(concept)
                    ))
        
        results.sort(key=lambda r: r.distance)
        
        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            if r.concept not in seen:
                seen.add(r.concept)
                unique.append(r)
        
        return unique[:n]
    
    def get_stats(self) -> Dict:
        """Get stack statistics."""
        return {
            "base": self.base.get_stats(),
            "domains": [d.get_stats() for d in self.domains],
            "context": self.context.get_stats(),
            "total_layers": 1 + len(self.domains) + 1,
            "total_pairs": sum(
                layer.corpus.pairs.__len__() 
                for layer in self.all_layers()
            ),
        }
    
    def print_stats(self):
        """Print stack statistics."""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print("CORPUS STACK")
        print(f"{'='*60}")
        print(f"Total layers: {stats['total_layers']}")
        print(f"Total pairs: {stats['total_pairs']}")
        print()
        print("BASE LAYER:")
        base = stats['base']
        print(f"  {base['name']}: {base['pairs']} pairs, "
              f"{base['dimensions']} dims, {base['concepts']} concepts")
        print()
        if stats['domains']:
            print("DOMAIN LAYERS:")
            for d in stats['domains']:
                print(f"  {d['name']}: {d['pairs']} pairs, "
                      f"{d['dimensions']} dims, {d['concepts']} concepts")
            print()
        print("CONTEXT LAYER:")
        ctx = stats['context']
        print(f"  Active mentions: {len(self.context._mentions)}")


class GeometricRAG:
    """
    RAG using geometric corpus layers instead of vector embeddings.
    
    Unlike traditional RAG:
    - We retrieve CONCEPTS, not text chunks
    - We preserve GEOMETRIC RELATIONSHIPS
    - We can TRAVERSE the space, not just retrieve
    """
    
    def __init__(self, stack: CorpusStack):
        self.stack = stack
    
    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve relevant concepts from all layers.
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of RetrievalResult
        """
        return self.stack.query(query, n=k)
    
    def retrieve_with_traversal(self, query: str, 
                                 traverse_dim: str = None,
                                 k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve and optionally traverse along a dimension.
        
        This enables queries like:
        - "What's more formal than X?"
        - "What's the opposite of Y?"
        """
        results = self.retrieve(query, k=k)
        
        if traverse_dim and results:
            # Traverse from top result
            top = results[0]
            traversed = self.stack.traverse(top.concept, traverse_dim)
            if traversed:
                # Add traversal result at top
                results.insert(0, RetrievalResult(
                    concept=traversed,
                    distance=0.0,  # Direct traversal
                    layer_name="traversal",
                ))
        
        return results
    
    def explain_relationship(self, concept1: str, concept2: str) -> Optional[str]:
        """
        Explain the geometric relationship between two concepts.
        
        Returns a description like:
        "king → queen: 1.62φ along gender dimension"
        """
        pos1 = None
        pos2 = None
        
        for layer in self.stack.all_layers():
            if pos1 is None:
                pos1 = layer.corpus.get_position(concept1)
            if pos2 is None:
                pos2 = layer.corpus.get_position(concept2)
        
        if pos1 is None or pos2 is None:
            return None
        
        # Compute delta
        delta = pos2 - pos1
        magnitude = np.linalg.norm(delta)
        
        # Find dominant dimension
        if len(delta) > 0:
            max_idx = np.argmax(np.abs(delta))
            for layer in self.stack.all_layers():
                for dim_name, dim in layer.corpus.dimensions.items():
                    if dim.index == max_idx:
                        return f"{concept1} → {concept2}: {magnitude:.2f}φ along {dim_name}"
        
        return f"{concept1} → {concept2}: {magnitude:.2f}φ"


# =============================================================================
# DEMO
# =============================================================================

def create_base_corpus() -> CorpusLayer:
    """Create a base corpus with language fundamentals."""
    base = CorpusLayer("base", priority=0)
    
    # Gender dimension
    base.add_pair("king", "queen", "gender")
    base.add_pair("man", "woman", "gender")
    base.add_pair("boy", "girl", "gender")
    base.add_pair("father", "mother", "gender")
    base.add_pair("brother", "sister", "gender")
    base.add_pair("prince", "princess", "gender")
    base.add_pair("actor", "actress", "gender")
    
    # Age dimension
    base.add_pair("boy", "man", "age")
    base.add_pair("girl", "woman", "age")
    base.add_pair("child", "adult", "age")
    base.add_pair("puppy", "dog", "age")
    base.add_pair("kitten", "cat", "age")
    base.add_pair("calf", "cow", "age")
    
    # Size dimension
    base.add_pair("large", "small", "size")
    base.add_pair("big", "little", "size")
    base.add_pair("giant", "tiny", "size")
    base.add_pair("house", "cottage", "size")
    base.add_pair("mansion", "house", "size")
    
    # Intensity dimension
    base.add_pair("hot", "warm", "intensity")
    base.add_pair("cold", "cool", "intensity")
    base.add_pair("loud", "quiet", "intensity")
    base.add_pair("fast", "slow", "intensity")
    
    # Formality dimension
    base.add_pair("hello", "hi", "formality")
    base.add_pair("goodbye", "bye", "formality")
    base.add_pair("please", "pls", "formality")
    base.add_pair("thank_you", "thanks", "formality")
    
    # Sentiment dimension
    base.add_pair("good", "bad", "sentiment")
    base.add_pair("happy", "sad", "sentiment")
    base.add_pair("love", "hate", "sentiment")
    base.add_pair("beautiful", "ugly", "sentiment")
    
    base.corpus.recompute()
    return base


def create_chess_domain() -> CorpusLayer:
    """Create a chess domain corpus."""
    chess = CorpusLayer("chess", priority=50)
    
    # Piece hierarchy (using shared "regality" concept)
    chess.add_pair("pawn", "knight", "piece_value")
    chess.add_pair("knight", "bishop", "piece_value")
    chess.add_pair("bishop", "rook", "piece_value")
    chess.add_pair("rook", "queen", "piece_value")
    chess.add_pair("queen", "king", "piece_value")
    
    # Promotion (age-like transformation)
    chess.add_pair("pawn", "queen", "promotion")
    chess.add_pair("pawn", "rook", "promotion")
    chess.add_pair("pawn", "bishop", "promotion")
    chess.add_pair("pawn", "knight", "promotion")
    
    # Movement types
    chess.add_pair("pawn", "knight", "movement_complexity")
    chess.add_pair("rook", "bishop", "movement_type")
    chess.add_pair("queen", "king", "movement_range")
    
    # Game phases
    chess.add_pair("opening", "middlegame", "game_phase")
    chess.add_pair("middlegame", "endgame", "game_phase")
    
    # Strategies
    chess.add_pair("attack", "defense", "strategy")
    chess.add_pair("aggressive", "passive", "strategy")
    chess.add_pair("tactical", "positional", "play_style")
    
    chess.corpus.recompute()
    return chess


def create_cooking_domain() -> CorpusLayer:
    """Create a cooking domain corpus."""
    cooking = CorpusLayer("cooking", priority=50)
    
    # Temperature (shared with base "intensity")
    cooking.add_pair("raw", "cooked", "cooking_state")
    cooking.add_pair("cold", "hot", "temperature")
    cooking.add_pair("simmer", "boil", "temperature")
    cooking.add_pair("warm", "sear", "temperature")
    
    # Cooking methods
    cooking.add_pair("boil", "steam", "method")
    cooking.add_pair("fry", "bake", "method")
    cooking.add_pair("grill", "roast", "method")
    cooking.add_pair("saute", "stir_fry", "method")
    
    # Texture
    cooking.add_pair("soft", "crispy", "texture")
    cooking.add_pair("tender", "tough", "texture")
    cooking.add_pair("moist", "dry", "texture")
    
    # Taste
    cooking.add_pair("sweet", "sour", "taste")
    cooking.add_pair("salty", "bland", "taste")
    cooking.add_pair("spicy", "mild", "taste")
    cooking.add_pair("bitter", "sweet", "taste")
    
    # Ingredients
    cooking.add_pair("flour", "bread", "transformation")
    cooking.add_pair("egg", "omelette", "transformation")
    cooking.add_pair("milk", "cheese", "transformation")
    
    cooking.corpus.recompute()
    return cooking


def demo_corpus_layers():
    """Demonstrate the corpus layer system."""
    print("=" * 60)
    print("DEMO: Corpus Layers - Attachable Knowledge System")
    print("=" * 60)
    print()
    
    # Create base corpus
    print("Step 1: Create base corpus (language fundamentals)")
    print("-" * 60)
    base = create_base_corpus()
    print(f"  Base corpus: {len(base.corpus.pairs)} pairs, "
          f"{len(base.corpus.dimensions)} dimensions")
    print()
    
    # Create stack
    print("Step 2: Create corpus stack")
    print("-" * 60)
    stack = CorpusStack(base)
    stack.print_stats()
    print()
    
    # Query base only
    print("Step 3: Query base corpus")
    print("-" * 60)
    results = stack.query("What is a king?")
    print(f"  Query: 'What is a king?'")
    print(f"  Results: {[r.concept for r in results[:5]]}")
    print()
    
    # Attach chess domain
    print("Step 4: Attach chess domain")
    print("-" * 60)
    chess = create_chess_domain()
    stack.attach_domain(chess)
    print(f"  Chess domain: {len(chess.corpus.pairs)} pairs, "
          f"{len(chess.corpus.dimensions)} dimensions")
    stack.print_stats()
    print()
    
    # Query with chess attached
    print("Step 5: Query with chess domain attached")
    print("-" * 60)
    results = stack.query("What is a king?")
    print(f"  Query: 'What is a king?'")
    print(f"  Results: {[(r.concept, r.layer_name) for r in results[:5]]}")
    print()
    
    results = stack.query("Tell me about the queen")
    print(f"  Query: 'Tell me about the queen'")
    print(f"  Results: {[(r.concept, r.layer_name) for r in results[:5]]}")
    print()
    
    # Traversal
    print("Step 6: Geometric traversal")
    print("-" * 60)
    rag = GeometricRAG(stack)
    
    explanation = rag.explain_relationship("king", "queen")
    print(f"  {explanation}")
    
    explanation = rag.explain_relationship("pawn", "queen")
    print(f"  {explanation}")
    print()
    
    # Detach chess, attach cooking
    print("Step 7: Detach chess, attach cooking")
    print("-" * 60)
    stack.detach_domain("chess")
    cooking = create_cooking_domain()
    stack.attach_domain(cooking)
    print(f"  Cooking domain: {len(cooking.corpus.pairs)} pairs")
    stack.print_stats()
    print()
    
    # Query with cooking
    print("Step 8: Query with cooking domain")
    print("-" * 60)
    results = stack.query("How do I cook something?")
    print(f"  Query: 'How do I cook something?'")
    print(f"  Results: {[(r.concept, r.layer_name) for r in results[:5]]}")
    print()
    
    # Context layer
    print("Step 9: Context layer (conversation state)")
    print("-" * 60)
    stack.context.add_mention("boil")
    stack.context.add_mention("temperature")
    stack.context.advance_turn()
    
    print(f"  Added mentions: boil, temperature")
    print(f"  Active concepts: {stack.context.get_active_concepts()}")
    
    # Query with context boost
    results = stack.query("What about hot?")
    print(f"  Query: 'What about hot?' (with context)")
    print(f"  Results: {[(r.concept, r.layer_name) for r in results[:5]]}")
    print()
    
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print()
    print("Key insights:")
    print("  1. Base corpus provides language fundamentals")
    print("  2. Domain corpora attach/detach for topic-specific knowledge")
    print("  3. Context layer tracks conversation state")
    print("  4. Shared dimensions are the hooks between layers")
    print("  5. Geometric traversal enables 'what's the opposite of X?'")
    print()
    
    return stack


if __name__ == "__main__":
    demo_corpus_layers()
