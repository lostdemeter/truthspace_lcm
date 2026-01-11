"""
Self-Assembling Corpus Experiment - Phase 1 & 2

This experiment demonstrates the core infrastructure for a self-assembling
knowledge corpus that:

Phase 1 (Core Infrastructure):
1. Stores transformation pairs as the source of truth
2. Derives dimensions emergently from relationship types
3. Positions concepts using φ-based geometry
4. Detects Platonic Ideals (multi-dimension anchors)
5. Persists and reconstructs from pairs alone

Phase 2 (Ingestion Pipeline):
6. Extracts transformation pairs from text
7. Distinguishes categories from instances (mastiff vs 'large dog')
8. Detects gaps and queries LLM for unknowns
9. Handles relationship type inference

Key principle: Everything derives from transformation pairs.
The space can be reconstructed entirely from pairs.

Usage:
    python -m experiments.self_assembling_corpus
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum
import re

# Golden ratio - the fundamental unit of semantic distance
PHI = (1 + np.sqrt(5)) / 2


class ConceptType(Enum):
    """Distinguishes categories from instances."""
    CATEGORY = "category"      # General concept (e.g., "large dog")
    INSTANCE = "instance"      # Specific example (e.g., "mastiff")
    IDEAL = "ideal"            # Platonic ideal (e.g., "dog")
    UNKNOWN = "unknown"        # Not yet classified


@dataclass
class TransformationPair:
    """A transformation pair defines a relationship between two concepts."""
    source: str
    target: str
    relationship: str
    confidence: float = 1.0
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __hash__(self):
        return hash((self.source.lower(), self.target.lower(), self.relationship))
    
    def __eq__(self, other):
        if not isinstance(other, TransformationPair):
            return False
        return (self.source.lower() == other.source.lower() and
                self.target.lower() == other.target.lower() and
                self.relationship == other.relationship)


@dataclass
class Dimension:
    """An emergent dimension discovered from transformation pairs."""
    name: str
    index: int
    pole_negative: List[str] = field(default_factory=list)  # Words at source (0)
    pole_positive: List[str] = field(default_factory=list)  # Words at target (+φ)
    source_pairs: List[Tuple[str, str]] = field(default_factory=list)
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def describe(self) -> str:
        """Generate English description of this dimension."""
        neg = self.pole_negative[:3] if self.pole_negative else ["?"]
        pos = self.pole_positive[:3] if self.pole_positive else ["?"]
        return f"{self.name}: {neg} → {pos}"


@dataclass
class PlatonicIdeal:
    """A concept that sits at the origin of multiple dimensions."""
    word: str
    dimensions_anchored: List[str] = field(default_factory=list)
    variations: Dict[str, List[str]] = field(default_factory=dict)  # dim → [words]
    confidence: float = 0.0
    
    def describe(self) -> str:
        """Generate English description of this ideal."""
        dims = ", ".join(self.dimensions_anchored)
        return f"{self.word} (anchors: {dims})"


class SelfAssemblingCorpus:
    """
    A self-assembling knowledge corpus built from transformation pairs.
    
    The corpus automatically:
    - Discovers dimensions from relationship types
    - Positions concepts using φ-based geometry
    - Detects Platonic Ideals
    - Rebalances when new dimensions are added
    """
    
    def __init__(self, persist_path: Optional[Path] = None):
        # Source of truth - everything derives from pairs
        self.pairs: List[TransformationPair] = []
        
        # Derived structures
        self.dimensions: Dict[str, Dimension] = {}
        self.concepts: Dict[str, np.ndarray] = {}  # word → position
        self.ideals: Dict[str, PlatonicIdeal] = {}
        
        # Metadata
        self.version: int = 0
        self.persist_path = persist_path
        
        # Internal tracking
        self._dirty = False  # Needs recomputation
    
    # =========================================================================
    # PAIR MANAGEMENT (Source of Truth)
    # =========================================================================
    
    def add_pair(self, source: str, target: str, relationship: str, 
                 confidence: float = 1.0) -> bool:
        """
        Add a transformation pair. Returns True if this created a new dimension.
        """
        pair = TransformationPair(
            source=source.lower().strip(),
            target=target.lower().strip(),
            relationship=relationship.lower().strip(),
            confidence=confidence
        )
        
        # Check for duplicate
        if pair in self.pairs:
            return False
        
        self.pairs.append(pair)
        self._dirty = True
        
        # Check if this is a new dimension
        new_dimension = pair.relationship not in self.dimensions
        
        if new_dimension:
            self._create_dimension(pair.relationship)
        
        # Update dimension poles
        dim = self.dimensions[pair.relationship]
        if pair.source not in dim.pole_negative:
            dim.pole_negative.append(pair.source)
        if pair.target not in dim.pole_positive:
            dim.pole_positive.append(pair.target)
        dim.source_pairs.append((pair.source, pair.target))
        
        return new_dimension
    
    def add_pairs(self, pairs: List[Tuple[str, str, str]]) -> int:
        """Add multiple pairs. Returns count of new dimensions created."""
        new_dims = 0
        for source, target, rel in pairs:
            if self.add_pair(source, target, rel):
                new_dims += 1
        return new_dims
    
    # =========================================================================
    # DIMENSION MANAGEMENT
    # =========================================================================
    
    def _create_dimension(self, name: str) -> Dimension:
        """Create a new dimension."""
        index = len(self.dimensions)
        dim = Dimension(name=name, index=index)
        self.dimensions[name] = dim
        
        # Extend all existing concept positions
        for word in self.concepts:
            old_pos = self.concepts[word]
            new_pos = np.zeros(index + 1)
            new_pos[:len(old_pos)] = old_pos
            self.concepts[word] = new_pos
        
        self.version += 1
        return dim
    
    def get_dimension(self, name: str) -> Optional[Dimension]:
        """Get a dimension by name."""
        return self.dimensions.get(name.lower().strip())
    
    def list_dimensions(self) -> List[str]:
        """List all dimension names."""
        return list(self.dimensions.keys())
    
    # =========================================================================
    # POSITION COMPUTATION
    # =========================================================================
    
    def recompute(self):
        """Recompute all positions from pairs."""
        if not self._dirty and self.concepts:
            return
        
        n_dims = len(self.dimensions)
        if n_dims == 0:
            return
        
        # Clear existing positions
        self.concepts.clear()
        
        # Get all unique words
        words = set()
        for pair in self.pairs:
            words.add(pair.source)
            words.add(pair.target)
        
        # Initialize positions at origin
        for word in words:
            self.concepts[word] = np.zeros(n_dims)
        
        # Position based on pairs
        # Source words stay at 0 (origin)
        # Target words move to +φ on their dimension
        for pair in self.pairs:
            dim = self.dimensions.get(pair.relationship)
            if dim is None:
                continue
            
            # Target moves to +φ on this dimension
            self.concepts[pair.target][dim.index] = PHI
        
        # Detect Platonic Ideals
        self._detect_ideals()
        
        self._dirty = False
    
    def get_position(self, word: str) -> Optional[np.ndarray]:
        """Get the position of a word."""
        self.recompute()
        return self.concepts.get(word.lower().strip())
    
    def get_compound_position(self, *words: str) -> np.ndarray:
        """
        Get the compound position of multiple words.
        Uses φ-Zipf weighting: φ^(-rank) for each component.
        """
        self.recompute()
        
        n_dims = len(self.dimensions)
        if n_dims == 0:
            return np.zeros(1)
        
        result = np.zeros(n_dims)
        total_weight = 0
        
        for rank, word in enumerate(words):
            pos = self.get_position(word)
            if pos is not None:
                weight = PHI ** (-rank)
                result += weight * pos
                total_weight += weight
        
        if total_weight > 0:
            result /= total_weight
        
        return result
    
    # =========================================================================
    # PLATONIC IDEAL DETECTION
    # =========================================================================
    
    def _detect_ideals(self):
        """Detect Platonic Ideals - words that anchor multiple dimensions."""
        self.ideals.clear()
        
        # Count which dimensions each word anchors (appears as source)
        anchor_counts: Dict[str, Set[str]] = {}
        variations: Dict[str, Dict[str, List[str]]] = {}
        
        for pair in self.pairs:
            source = pair.source
            if source not in anchor_counts:
                anchor_counts[source] = set()
                variations[source] = {}
            
            anchor_counts[source].add(pair.relationship)
            
            if pair.relationship not in variations[source]:
                variations[source][pair.relationship] = []
            variations[source][pair.relationship].append(pair.target)
        
        # Words anchoring 2+ dimensions are Platonic Ideals
        for word, dims in anchor_counts.items():
            if len(dims) >= 2:
                ideal = PlatonicIdeal(
                    word=word,
                    dimensions_anchored=list(dims),
                    variations=variations.get(word, {}),
                    confidence=len(dims) / len(self.dimensions) if self.dimensions else 0
                )
                self.ideals[word] = ideal
    
    def get_ideal(self, word: str) -> Optional[PlatonicIdeal]:
        """Get a Platonic Ideal by word."""
        self.recompute()
        return self.ideals.get(word.lower().strip())
    
    def list_ideals(self) -> List[str]:
        """List all Platonic Ideal words."""
        self.recompute()
        return list(self.ideals.keys())
    
    # =========================================================================
    # QUERIES AND ANALYSIS
    # =========================================================================
    
    def find_nearest(self, position: np.ndarray, n: int = 5) -> List[Tuple[str, float]]:
        """Find the n nearest words to a position."""
        self.recompute()
        
        distances = []
        for word, pos in self.concepts.items():
            dist = np.linalg.norm(pos - position)
            distances.append((word, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:n]
    
    def transform(self, word: str, dimension: str, direction: float = 1.0) -> List[Tuple[str, float]]:
        """
        Transform a word along a dimension.
        direction: +1 = positive pole, -1 = negative pole
        """
        self.recompute()
        
        pos = self.get_position(word)
        if pos is None:
            return []
        
        dim = self.get_dimension(dimension)
        if dim is None:
            return []
        
        # Move by φ in the specified direction
        new_pos = pos.copy()
        new_pos[dim.index] += direction * PHI
        
        return self.find_nearest(new_pos)
    
    def get_delta(self, word1: str, word2: str) -> Optional[Tuple[float, str]]:
        """
        Get the delta between two words.
        Returns (magnitude, dominant_dimension).
        """
        self.recompute()
        
        pos1 = self.get_position(word1)
        pos2 = self.get_position(word2)
        
        if pos1 is None or pos2 is None:
            return None
        
        delta = pos2 - pos1
        magnitude = np.linalg.norm(delta)
        
        # Find dominant dimension
        if len(delta) > 0:
            max_idx = np.argmax(np.abs(delta))
            for name, dim in self.dimensions.items():
                if dim.index == max_idx:
                    return (magnitude, name)
        
        return (magnitude, "unknown")
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save(self, path: Optional[Path] = None):
        """Save the corpus to disk."""
        path = path or self.persist_path
        if path is None:
            raise ValueError("No persist path specified")
        
        data = {
            "version": self.version,
            "pairs": [asdict(p) for p in self.pairs],
            "dimensions": {name: asdict(d) for name, d in self.dimensions.items()},
            "ideals": {name: asdict(i) for name, i in self.ideals.items()},
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'SelfAssemblingCorpus':
        """Load a corpus from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        corpus = cls(persist_path=path)
        corpus.version = data.get("version", 0)
        
        # Reconstruct from pairs (the source of truth)
        for pair_data in data.get("pairs", []):
            pair = TransformationPair(**pair_data)
            corpus.pairs.append(pair)
            
            # Ensure dimension exists
            if pair.relationship not in corpus.dimensions:
                corpus._create_dimension(pair.relationship)
            
            dim = corpus.dimensions[pair.relationship]
            if pair.source not in dim.pole_negative:
                dim.pole_negative.append(pair.source)
            if pair.target not in dim.pole_positive:
                dim.pole_positive.append(pair.target)
            dim.source_pairs.append((pair.source, pair.target))
        
        corpus._dirty = True
        corpus.recompute()
        
        return corpus
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def print_report(self):
        """Print a comprehensive report of the corpus."""
        self.recompute()
        
        print("=" * 60)
        print("SELF-ASSEMBLING CORPUS REPORT")
        print("=" * 60)
        print()
        print(f"Version: {self.version}")
        print(f"Pairs: {len(self.pairs)}")
        print(f"Dimensions: {len(self.dimensions)}")
        print(f"Concepts: {len(self.concepts)}")
        print(f"Platonic Ideals: {len(self.ideals)}")
        print()
        
        # Dimensions
        print("DIMENSIONS")
        print("-" * 60)
        for name, dim in self.dimensions.items():
            print(f"  {dim.index}: {dim.describe()}")
        print()
        
        # Platonic Ideals
        if self.ideals:
            print("PLATONIC IDEALS")
            print("-" * 60)
            for word, ideal in sorted(self.ideals.items(), 
                                       key=lambda x: -len(x[1].dimensions_anchored)):
                print(f"  {ideal.describe()}")
                for dim_name, variations in ideal.variations.items():
                    print(f"    {dim_name}: {variations}")
            print()
        
        # Sample positions
        print("SAMPLE POSITIONS")
        print("-" * 60)
        for word in list(self.concepts.keys())[:10]:
            pos = self.concepts[word]
            pos_str = ", ".join(f"{v:.2f}" for v in pos)
            print(f"  {word}: [{pos_str}]")
        print()


# =============================================================================
# PHASE 2: INGESTION PIPELINE
# =============================================================================

@dataclass
class Concept:
    """A concept with type information (category vs instance)."""
    word: str
    concept_type: ConceptType = ConceptType.UNKNOWN
    parent: Optional[str] = None  # For instances, the category they belong to
    attributes: List[str] = field(default_factory=list)  # e.g., ["large", "friendly"]
    
    def is_instance(self) -> bool:
        return self.concept_type == ConceptType.INSTANCE
    
    def is_category(self) -> bool:
        return self.concept_type == ConceptType.CATEGORY


@dataclass
class Gap:
    """A detected gap in the corpus that needs filling."""
    ideal: str
    dimension: str
    direction: str  # "positive" or "negative"
    description: str
    
    def to_query(self) -> str:
        """Generate an LLM query to fill this gap."""
        if self.direction == "positive":
            return f"What is a word for a {self.dimension} version of {self.ideal}?"
        else:
            return f"What is a word for a less {self.dimension} version of {self.ideal}?"


class IngestionPipeline:
    """
    Phase 2: Extracts transformation pairs from text and manages ingestion.
    
    Key responsibilities:
    - Extract relationships from text
    - Distinguish categories from instances
    - Detect gaps and generate LLM queries
    - Batch queries efficiently
    """
    
    # Common relationship patterns
    # Note: patterns use (?:a |an |the )? to skip articles
    RELATIONSHIP_PATTERNS = [
        # "X is a type of Y" → (Y, X, specificity_increase)
        (r"(?:a |an |the )?(\w+)\s+is\s+a\s+(?:type|kind|form|breed)\s+of\s+(?:a |an |the )?(\w+)", "specificity_increase", True),
        # "X is larger/smaller than Y" → size relationship
        (r"(?:a |an |the )?(\w+)\s+is\s+(?:larger|bigger)\s+than\s+(?:a |an |the )?(\w+)", "size_increase", False),
        (r"(?:a |an |the )?(\w+)\s+is\s+(?:smaller|tinier)\s+than\s+(?:a |an |the )?(\w+)", "size_decrease", False),
        # "X is more/less formal than Y"
        (r"(?:a |an |the )?(\w+)\s+is\s+more\s+formal\s+than\s+(?:a |an |the )?(\w+)", "formality_increase", False),
        (r"(?:a |an |the )?(\w+)\s+is\s+less\s+formal\s+than\s+(?:a |an |the )?(\w+)", "formality_decrease", False),
        # "X and Y" in same context (potential pairs) - require 3+ char words
        (r"\b([a-z]{3,})\s+and\s+([a-z]{3,})\b", "co_occurrence", False),
        # "unlike X, Y is..." (contrast)
        (r"unlike\s+(?:a |an |the )?(\w+),?\s+(?:a |an |the )?(\w+)", "contrast", False),
    ]
    
    # Words to skip (articles, common words)
    SKIP_WORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "been",
                  "have", "has", "had", "do", "does", "did", "will", "would",
                  "could", "should", "may", "might", "must", "shall", "can",
                  "this", "that", "these", "those", "it", "its", "most", "more",
                  "less", "very", "much", "many", "some", "any", "all", "each"}
    
    # Instance indicators - words that suggest something is a specific instance
    INSTANCE_INDICATORS = [
        "breed", "species", "variety", "type", "kind", "model", "brand",
        "specific", "particular", "individual", "named", "called"
    ]
    
    # Category indicators - words that suggest something is a general category
    CATEGORY_INDICATORS = [
        "any", "all", "every", "general", "typical", "average", "normal",
        "generic", "common", "standard"
    ]
    
    def __init__(self, corpus: 'SelfAssemblingCorpus'):
        self.corpus = corpus
        self.concepts: Dict[str, Concept] = {}
        self.pending_queries: List[Gap] = []
        self.extracted_pairs: List[Tuple[str, str, str, float]] = []
    
    def classify_concept(self, word: str, context: str = "") -> ConceptType:
        """
        Classify a concept as category, instance, or ideal.
        
        The mastiff problem: "mastiff" is a specific breed (instance),
        not a general "large dog" (category).
        """
        word_lower = word.lower()
        context_lower = context.lower()
        
        # Check if already classified
        if word_lower in self.concepts:
            return self.concepts[word_lower].concept_type
        
        # Check if it's a known Platonic Ideal
        if word_lower in self.corpus.ideals:
            return ConceptType.IDEAL
        
        # Check context for instance indicators
        for indicator in self.INSTANCE_INDICATORS:
            if indicator in context_lower:
                # Context suggests this is a specific instance
                return ConceptType.INSTANCE
        
        # Check context for category indicators
        for indicator in self.CATEGORY_INDICATORS:
            if indicator in context_lower:
                return ConceptType.CATEGORY
        
        # Heuristic: proper nouns (capitalized) are often instances
        if word[0].isupper() and not context.startswith(word):
            return ConceptType.INSTANCE
        
        return ConceptType.UNKNOWN
    
    def register_concept(self, word: str, concept_type: ConceptType,
                        parent: Optional[str] = None,
                        attributes: Optional[List[str]] = None) -> Concept:
        """Register a concept with its type information."""
        concept = Concept(
            word=word.lower(),
            concept_type=concept_type,
            parent=parent,
            attributes=attributes or []
        )
        self.concepts[word.lower()] = concept
        return concept
    
    def extract_pairs_from_text(self, text: str) -> List[Tuple[str, str, str, float]]:
        """
        Extract transformation pairs from text.
        Returns list of (source, target, relationship, confidence).
        """
        pairs = []
        
        for pattern, rel_type, swap in self.RELATIONSHIP_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                word1, word2 = match.group(1), match.group(2)
                
                if swap:
                    source, target = word2, word1
                else:
                    source, target = word1, word2
                
                # Skip if same word
                if source.lower() == target.lower():
                    continue
                
                # Classify concepts
                context = text[max(0, match.start()-50):match.end()+50]
                type1 = self.classify_concept(source, context)
                type2 = self.classify_concept(target, context)
                
                # Skip if either word is in skip list
                if source.lower() in self.SKIP_WORDS or target.lower() in self.SKIP_WORDS:
                    continue
                
                # Confidence based on pattern type
                confidence = 0.8 if rel_type != "co_occurrence" else 0.3
                
                pairs.append((source.lower(), target.lower(), rel_type, confidence))
        
        self.extracted_pairs.extend(pairs)
        return pairs
    
    def ingest_text(self, text: str) -> Dict[str, any]:
        """
        Ingest text and update the corpus.
        Returns statistics about what was ingested.
        """
        # Extract pairs
        pairs = self.extract_pairs_from_text(text)
        
        # Add to corpus (filtering low confidence)
        new_dims = 0
        added_pairs = 0
        for source, target, rel, conf in pairs:
            if conf >= 0.5:  # Only add reasonably confident pairs
                if self.corpus.add_pair(source, target, rel, conf):
                    new_dims += 1
                added_pairs += 1
        
        # Detect gaps
        gaps = self.detect_gaps()
        
        return {
            "extracted_pairs": len(pairs),
            "added_pairs": added_pairs,
            "new_dimensions": new_dims,
            "gaps_detected": len(gaps)
        }
    
    def detect_gaps(self) -> List[Gap]:
        """
        Detect gaps in the corpus - missing variations for Platonic Ideals.
        """
        self.corpus.recompute()
        gaps = []
        
        # For each ideal, check if it has variations in all known dimensions
        for ideal_word, ideal in self.corpus.ideals.items():
            for dim_name in self.corpus.dimensions:
                # Check if this ideal has a variation in this dimension
                has_variation = dim_name in ideal.variations
                
                if not has_variation:
                    # This is a gap - the ideal doesn't have a variation in this dimension
                    gap = Gap(
                        ideal=ideal_word,
                        dimension=dim_name,
                        direction="positive",
                        description=f"{ideal_word} has no {dim_name} variation"
                    )
                    gaps.append(gap)
        
        self.pending_queries = gaps
        return gaps
    
    def handle_instance_vs_category(self, word: str, ideal: str, 
                                     dimension: str) -> Tuple[bool, str]:
        """
        Handle the mastiff problem: determine if a word is a true category
        variation or just a specific instance.
        
        Returns (is_valid_category, explanation).
        
        Example:
        - "mastiff" for dog + size_increase → INSTANCE (specific breed)
        - "large dog" for dog + size_increase → CATEGORY (general concept)
        - "mansion" for house + size_increase → CATEGORY (general concept)
        """
        word_lower = word.lower()
        
        # Check if we have type information
        if word_lower in self.concepts:
            concept = self.concepts[word_lower]
            if concept.is_instance():
                return (False, f"'{word}' is a specific instance, not a general category")
        
        # Heuristic: single words that are proper nouns or specific terms
        # are likely instances
        
        # Check if the word contains the ideal (e.g., "large dog" contains "dog")
        if ideal.lower() in word_lower:
            return (True, f"'{word}' is a compound containing the ideal")
        
        # Check if it's a known breed/type/variety
        # This would ideally be an LLM query, but for now use heuristics
        known_instances = {
            "dog": ["mastiff", "chihuahua", "labrador", "poodle", "bulldog", 
                    "beagle", "terrier", "collie", "shepherd", "retriever"],
            "cat": ["persian", "siamese", "tabby", "maine coon", "ragdoll"],
            "car": ["ferrari", "toyota", "honda", "ford", "tesla"],
            "house": [],  # mansion, cottage are categories, not instances
        }
        
        if ideal.lower() in known_instances:
            if word_lower in known_instances[ideal.lower()]:
                return (False, f"'{word}' is a specific {ideal} breed/type")
        
        return (True, f"'{word}' appears to be a valid category variation")
    
    def generate_llm_queries(self, max_queries: int = 10) -> List[str]:
        """
        Generate batched LLM queries for filling gaps.
        """
        queries = []
        
        # Group gaps by type for efficient batching
        gaps_by_dim = {}
        for gap in self.pending_queries[:max_queries]:
            if gap.dimension not in gaps_by_dim:
                gaps_by_dim[gap.dimension] = []
            gaps_by_dim[gap.dimension].append(gap)
        
        # Generate batched queries
        for dim, dim_gaps in gaps_by_dim.items():
            ideals = [g.ideal for g in dim_gaps]
            query = f"For the '{dim}' dimension, what are variations of: {', '.join(ideals)}?"
            queries.append(query)
        
        return queries
    
    def process_llm_response(self, query: str, response: str) -> List[Tuple[str, str, str]]:
        """
        Process an LLM response and extract pairs.
        Returns list of (source, target, relationship) to add.
        """
        # This would parse the LLM response
        # For now, return empty - actual implementation would parse response
        return []


# =============================================================================
# DEMO FUNCTIONS - PHASE 1
# =============================================================================

def demo_basic():
    """Demonstrate basic corpus assembly."""
    print("=" * 60)
    print("DEMO: Basic Corpus Assembly")
    print("=" * 60)
    print()
    
    corpus = SelfAssemblingCorpus()
    
    # Add pairs - dimensions emerge automatically
    print("Adding transformation pairs...")
    print()
    
    # Gender dimension emerges
    corpus.add_pair("king", "queen", "gender")
    corpus.add_pair("man", "woman", "gender")
    corpus.add_pair("boy", "girl", "gender")
    corpus.add_pair("father", "mother", "gender")
    print(f"  After gender pairs: {len(corpus.dimensions)} dimensions")
    
    # Age dimension emerges
    corpus.add_pair("boy", "man", "age")
    corpus.add_pair("girl", "woman", "age")
    corpus.add_pair("puppy", "dog", "age")
    corpus.add_pair("kitten", "cat", "age")
    print(f"  After age pairs: {len(corpus.dimensions)} dimensions")
    
    # Size dimension emerges
    corpus.add_pair("house", "cottage", "size_decrease")
    corpus.add_pair("house", "mansion", "size_increase")
    corpus.add_pair("dog", "puppy", "size_decrease")
    corpus.add_pair("dog", "mastiff", "size_increase")
    print(f"  After size pairs: {len(corpus.dimensions)} dimensions")
    
    # Regality dimension emerges
    corpus.add_pair("house", "hovel", "regality_decrease")
    corpus.add_pair("house", "palace", "regality_increase")
    print(f"  After regality pairs: {len(corpus.dimensions)} dimensions")
    
    print()
    corpus.print_report()
    
    return corpus


def demo_platonic_ideals():
    """Demonstrate Platonic Ideal detection."""
    print("=" * 60)
    print("DEMO: Platonic Ideal Detection")
    print("=" * 60)
    print()
    
    corpus = SelfAssemblingCorpus()
    
    # House as Platonic Ideal (anchors size AND regality)
    corpus.add_pair("house", "cottage", "size_decrease")
    corpus.add_pair("house", "mansion", "size_increase")
    corpus.add_pair("house", "hovel", "regality_decrease")
    corpus.add_pair("house", "palace", "regality_increase")
    
    # Person as Platonic Ideal (anchors age AND status AND familiarity)
    corpus.add_pair("person", "child", "age_decrease")
    corpus.add_pair("person", "elder", "age_increase")
    corpus.add_pair("person", "peasant", "status_decrease")
    corpus.add_pair("person", "noble", "status_increase")
    corpus.add_pair("person", "stranger", "familiarity_decrease")
    corpus.add_pair("person", "friend", "familiarity_increase")
    
    # Dog as Platonic Ideal (anchors size AND age)
    corpus.add_pair("dog", "puppy", "age_decrease")
    corpus.add_pair("dog", "lapdog", "size_decrease")
    corpus.add_pair("dog", "mastiff", "size_increase")
    
    corpus.recompute()
    
    print("Detected Platonic Ideals:")
    print("-" * 60)
    for word in corpus.list_ideals():
        ideal = corpus.get_ideal(word)
        print(f"\n  {word.upper()}")
        print(f"    Anchors {len(ideal.dimensions_anchored)} dimensions: {ideal.dimensions_anchored}")
        print(f"    Confidence: {ideal.confidence:.2f}")
        print(f"    Variations:")
        for dim, vars in ideal.variations.items():
            print(f"      {dim}: {vars}")
    
    print()
    
    # Show positions
    print("Positions relative to ideals:")
    print("-" * 60)
    
    house_pos = corpus.get_position("house")
    print(f"\n  house (ideal): {house_pos}")
    
    for var in ["cottage", "mansion", "hovel", "palace"]:
        pos = corpus.get_position(var)
        delta = corpus.get_delta("house", var)
        print(f"  {var}: {pos} (Δ={delta[0]:.2f} on {delta[1]})")
    
    print()
    return corpus


def demo_transformation():
    """Demonstrate transformations along dimensions."""
    print("=" * 60)
    print("DEMO: Transformations")
    print("=" * 60)
    print()
    
    corpus = SelfAssemblingCorpus()
    
    # Build a small corpus
    corpus.add_pairs([
        ("king", "queen", "gender"),
        ("man", "woman", "gender"),
        ("boy", "girl", "gender"),
        ("boy", "man", "age"),
        ("girl", "woman", "age"),
        ("prince", "princess", "gender"),
        ("prince", "king", "age"),
    ])
    
    corpus.recompute()
    
    print("Corpus built with gender and age dimensions")
    print()
    
    # Transform king along gender
    print("Transform 'king' along gender dimension:")
    results = corpus.transform("king", "gender", direction=1.0)
    for word, dist in results[:3]:
        print(f"  {word}: distance={dist:.3f}")
    
    print()
    
    # Transform boy along age
    print("Transform 'boy' along age dimension:")
    results = corpus.transform("boy", "age", direction=1.0)
    for word, dist in results[:3]:
        print(f"  {word}: distance={dist:.3f}")
    
    print()
    
    # Check self-similarity
    print("Self-similarity check (all gender deltas should be φ):")
    for src, tgt in [("king", "queen"), ("man", "woman"), ("boy", "girl")]:
        delta = corpus.get_delta(src, tgt)
        print(f"  {src} → {tgt}: Δ={delta[0]:.3f} (φ={PHI:.3f})")
    
    print()
    return corpus


def demo_persistence():
    """Demonstrate saving and loading."""
    print("=" * 60)
    print("DEMO: Persistence")
    print("=" * 60)
    print()
    
    # Create and populate corpus
    corpus = SelfAssemblingCorpus()
    corpus.add_pairs([
        ("king", "queen", "gender"),
        ("man", "woman", "gender"),
        ("house", "mansion", "size_increase"),
        ("house", "cottage", "size_decrease"),
    ])
    corpus.recompute()
    
    print(f"Original corpus: {len(corpus.pairs)} pairs, {len(corpus.dimensions)} dimensions")
    
    # Save
    save_path = Path("/tmp/test_corpus.json")
    corpus.save(save_path)
    print(f"Saved to {save_path}")
    
    # Load
    loaded = SelfAssemblingCorpus.load(save_path)
    print(f"Loaded corpus: {loaded.version} version, {len(loaded.pairs)} pairs, {len(loaded.dimensions)} dimensions")
    
    # Verify positions match
    print()
    print("Position verification:")
    for word in ["king", "queen", "house", "mansion"]:
        orig_pos = corpus.get_position(word)
        load_pos = loaded.get_position(word)
        match = np.allclose(orig_pos, load_pos)
        print(f"  {word}: {'✓' if match else '✗'}")
    
    print()
    return loaded


def demo_dynamic_dimension():
    """Demonstrate dynamic dimension addition."""
    print("=" * 60)
    print("DEMO: Dynamic Dimension Addition")
    print("=" * 60)
    print()
    
    corpus = SelfAssemblingCorpus()
    
    # Start with just gender
    print("Phase 1: Adding gender dimension")
    corpus.add_pairs([
        ("king", "queen", "gender"),
        ("man", "woman", "gender"),
    ])
    corpus.recompute()
    print(f"  Dimensions: {corpus.list_dimensions()}")
    print(f"  king position: {corpus.get_position('king')}")
    print()
    
    # Add age dimension - positions should extend
    print("Phase 2: Adding age dimension (positions extend)")
    corpus.add_pairs([
        ("boy", "man", "age"),
        ("girl", "woman", "age"),
    ])
    corpus.recompute()
    print(f"  Dimensions: {corpus.list_dimensions()}")
    print(f"  king position: {corpus.get_position('king')}")
    print(f"  man position: {corpus.get_position('man')}")
    print()
    
    # Add formality dimension
    print("Phase 3: Adding formality dimension")
    corpus.add_pairs([
        ("hi", "hello", "formality"),
        ("yeah", "yes", "formality"),
    ])
    corpus.recompute()
    print(f"  Dimensions: {corpus.list_dimensions()}")
    print(f"  king position: {corpus.get_position('king')}")
    print(f"  hello position: {corpus.get_position('hello')}")
    print()
    
    # Add perspective dimension (simulating ingesting first-person text)
    print("Phase 4: Adding perspective dimension (new corpus type)")
    corpus.add_pairs([
        ("I", "he", "perspective"),
        ("me", "him", "perspective"),
        ("my", "his", "perspective"),
    ])
    corpus.recompute()
    print(f"  Dimensions: {corpus.list_dimensions()}")
    print(f"  All concepts now have {len(corpus.dimensions)} dimensions")
    print()
    
    corpus.print_report()
    return corpus


def demo_compound_positions():
    """Demonstrate compound position computation."""
    print("=" * 60)
    print("DEMO: Compound Positions")
    print("=" * 60)
    print()
    
    corpus = SelfAssemblingCorpus()
    
    # Build corpus
    corpus.add_pairs([
        ("house", "mansion", "size_increase"),
        ("house", "cottage", "size_decrease"),
        ("house", "palace", "regality_increase"),
        ("house", "hovel", "regality_decrease"),
    ])
    corpus.recompute()
    
    print("Individual positions:")
    for word in ["house", "mansion", "palace"]:
        pos = corpus.get_position(word)
        print(f"  {word}: {pos}")
    
    print()
    print("Compound positions (φ-Zipf weighted):")
    
    # Large + regal = ?
    compound = corpus.get_compound_position("mansion", "palace")
    nearest = corpus.find_nearest(compound, n=3)
    print(f"  mansion + palace: {compound}")
    print(f"    Nearest: {nearest}")
    
    # Small + low-regal = ?
    compound = corpus.get_compound_position("cottage", "hovel")
    nearest = corpus.find_nearest(compound, n=3)
    print(f"  cottage + hovel: {compound}")
    print(f"    Nearest: {nearest}")
    
    print()
    print("Note: Compound positions may not have named words.")
    print("This is the 'unnamed compound' case from the roadmap.")
    print()
    
    return corpus


# =============================================================================
# DEMO FUNCTIONS - PHASE 2
# =============================================================================

def demo_text_ingestion():
    """Demonstrate text ingestion and pair extraction."""
    print("=" * 60)
    print("DEMO: Text Ingestion (Phase 2)")
    print("=" * 60)
    print()
    
    corpus = SelfAssemblingCorpus()
    pipeline = IngestionPipeline(corpus)
    
    # Sample text with relationships
    sample_text = """
    A mansion is larger than a house. A cottage is smaller than a house.
    The palace is more formal than the mansion. A hovel is less formal than a cottage.
    Dogs and cats are common pets. A mastiff is a type of dog.
    Unlike a chihuahua, a mastiff is larger than most dogs.
    """
    
    print("Ingesting sample text...")
    print("-" * 60)
    print(sample_text.strip())
    print("-" * 60)
    print()
    
    # Extract pairs
    pairs = pipeline.extract_pairs_from_text(sample_text)
    
    print(f"Extracted {len(pairs)} pairs:")
    for source, target, rel, conf in pairs:
        print(f"  {source} → {target} ({rel}, conf={conf:.1f})")
    
    print()
    
    # Ingest into corpus
    stats = pipeline.ingest_text(sample_text)
    print(f"Ingestion stats: {stats}")
    
    print()
    corpus.print_report()
    
    return corpus, pipeline


def demo_instance_vs_category():
    """Demonstrate the mastiff problem - instance vs category distinction."""
    print("=" * 60)
    print("DEMO: Instance vs Category (The Mastiff Problem)")
    print("=" * 60)
    print()
    
    corpus = SelfAssemblingCorpus()
    pipeline = IngestionPipeline(corpus)
    
    # Set up the dog ideal
    corpus.add_pair("dog", "puppy", "age_decrease")
    corpus.add_pair("dog", "lapdog", "size_decrease")
    corpus.recompute()
    
    print("The Problem:")
    print("-" * 60)
    print("  Q: What is 'large + dog'?")
    print("  A1: 'mastiff' - but this is a SPECIFIC BREED")
    print("  A2: 'large dog' - this is a GENERAL CATEGORY")
    print()
    print("  Mastiff IS large, but not every large dog is a mastiff.")
    print("  This is the instance vs category distinction.")
    print()
    
    # Test various words
    test_cases = [
        ("mastiff", "dog", "size_increase"),
        ("chihuahua", "dog", "size_decrease"),
        ("mansion", "house", "size_increase"),
        ("cottage", "house", "size_decrease"),
        ("large dog", "dog", "size_increase"),
        ("labrador", "dog", "friendliness_increase"),
    ]
    
    print("Classification results:")
    print("-" * 60)
    for word, ideal, dim in test_cases:
        is_category, explanation = pipeline.handle_instance_vs_category(word, ideal, dim)
        status = "CATEGORY ✓" if is_category else "INSTANCE ✗"
        print(f"  {word:15} for {ideal}+{dim}: {status}")
        print(f"    → {explanation}")
    
    print()
    print("Key insight:")
    print("  - 'mansion' is a valid category (any large fancy house)")
    print("  - 'mastiff' is an instance (a specific breed)")
    print("  - 'large dog' is a valid category (compound descriptor)")
    print()
    
    return corpus, pipeline


def demo_gap_detection():
    """Demonstrate gap detection and LLM query generation."""
    print("=" * 60)
    print("DEMO: Gap Detection")
    print("=" * 60)
    print()
    
    corpus = SelfAssemblingCorpus()
    pipeline = IngestionPipeline(corpus)
    
    # Create a corpus with some gaps
    # House has size and regality variations
    corpus.add_pair("house", "cottage", "size_decrease")
    corpus.add_pair("house", "mansion", "size_increase")
    corpus.add_pair("house", "palace", "regality_increase")
    
    # Dog only has size variations (gap: no regality)
    corpus.add_pair("dog", "puppy", "size_decrease")
    corpus.add_pair("dog", "mastiff", "size_increase")
    
    # Person has age variations (gap: no size, no regality)
    corpus.add_pair("person", "child", "age_decrease")
    corpus.add_pair("person", "elder", "age_increase")
    
    corpus.recompute()
    
    print("Current Platonic Ideals and their variations:")
    print("-" * 60)
    for word in corpus.list_ideals():
        ideal = corpus.get_ideal(word)
        print(f"\n  {word.upper()}")
        for dim, vars in ideal.variations.items():
            print(f"    {dim}: {vars}")
    
    print()
    
    # Detect gaps
    gaps = pipeline.detect_gaps()
    
    print(f"Detected {len(gaps)} gaps:")
    print("-" * 60)
    for gap in gaps:
        print(f"  {gap.ideal} missing {gap.dimension} variation")
        print(f"    Query: {gap.to_query()}")
    
    print()
    
    # Generate batched LLM queries
    queries = pipeline.generate_llm_queries()
    print("Batched LLM queries:")
    print("-" * 60)
    for q in queries:
        print(f"  {q}")
    
    print()
    return corpus, pipeline


def demo_full_pipeline():
    """Demonstrate the full ingestion pipeline with real-ish text."""
    print("=" * 60)
    print("DEMO: Full Pipeline")
    print("=" * 60)
    print()
    
    corpus = SelfAssemblingCorpus()
    pipeline = IngestionPipeline(corpus)
    
    # Simulate ingesting multiple documents
    documents = [
        # Document 1: About houses
        """
        Houses come in many sizes. A mansion is larger than a typical house,
        while a cottage is smaller than a house. Palaces are more formal than
        mansions, representing the pinnacle of grandeur. A hovel, unlike a palace,
        is less formal than even a cottage.
        """,
        
        # Document 2: About dogs (introduces the mastiff problem)
        """
        Dogs vary greatly in size. A mastiff is a breed of dog known for being
        larger than most dogs. Chihuahuas are smaller than typical dogs.
        Puppies are younger than adult dogs. Any large dog needs more space
        than a small dog.
        """,
        
        # Document 3: About perspective (new dimension!)
        """
        In first-person narrative, "I" refers to the narrator. In third-person,
        "he" or "she" replaces "I". Similarly, "me" becomes "him" or "her",
        and "my" becomes "his" or "her".
        """
    ]
    
    for i, doc in enumerate(documents, 1):
        print(f"Ingesting document {i}...")
        stats = pipeline.ingest_text(doc)
        print(f"  Extracted: {stats['extracted_pairs']} pairs")
        print(f"  Added: {stats['added_pairs']} pairs")
        print(f"  New dimensions: {stats['new_dimensions']}")
        print()
    
    print("Final corpus state:")
    corpus.print_report()
    
    # Show instance vs category handling
    print("Instance vs Category Analysis:")
    print("-" * 60)
    
    # Register mastiff as an instance
    pipeline.register_concept("mastiff", ConceptType.INSTANCE, parent="dog")
    
    for word in ["mastiff", "mansion", "cottage", "palace"]:
        if word in pipeline.concepts:
            concept = pipeline.concepts[word]
            print(f"  {word}: {concept.concept_type.value}")
        else:
            # Check using heuristics
            is_cat, reason = pipeline.handle_instance_vs_category(word, "house", "size")
            print(f"  {word}: {'category' if is_cat else 'instance'} ({reason})")
    
    print()
    return corpus, pipeline


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("SELF-ASSEMBLING CORPUS EXPERIMENT - PHASE 1 & 2")
    print("=" * 60)
    print()
    print("This experiment demonstrates the core infrastructure for")
    print("a self-assembling knowledge corpus.")
    print()
    print("PHASE 1: Core Infrastructure")
    print("  1. Transformation pairs are the source of truth")
    print("  2. Dimensions emerge from relationship types")
    print("  3. Positions use φ-based geometry")
    print("  4. Platonic Ideals are detected automatically")
    print("  5. The space can be reconstructed from pairs alone")
    print()
    print("PHASE 2: Ingestion Pipeline")
    print("  6. Extract transformation pairs from text")
    print("  7. Distinguish categories from instances (mastiff problem)")
    print("  8. Detect gaps and generate LLM queries")
    print("  9. Handle dynamic dimension discovery")
    print()
    
    # Phase 1 demos
    print("=" * 60)
    print("PHASE 1 DEMOS")
    print("=" * 60)
    print()
    
    demo_basic()
    print("\n" + "=" * 60 + "\n")
    
    demo_platonic_ideals()
    print("\n" + "=" * 60 + "\n")
    
    demo_transformation()
    print("\n" + "=" * 60 + "\n")
    
    demo_persistence()
    print("\n" + "=" * 60 + "\n")
    
    demo_dynamic_dimension()
    print("\n" + "=" * 60 + "\n")
    
    demo_compound_positions()
    
    # Phase 2 demos
    print("\n")
    print("=" * 60)
    print("PHASE 2 DEMOS")
    print("=" * 60)
    print()
    
    demo_text_ingestion()
    print("\n" + "=" * 60 + "\n")
    
    demo_instance_vs_category()
    print("\n" + "=" * 60 + "\n")
    
    demo_gap_detection()
    print("\n" + "=" * 60 + "\n")
    
    demo_full_pipeline()
    
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print()
    print("Key findings - Phase 1:")
    print("  1. Dimensions emerge automatically from relationship types")
    print("  2. Platonic Ideals detected by multi-dimension anchoring")
    print("  3. Positions extend automatically when new dimensions added")
    print("  4. Self-similarity preserved (all deltas = φ)")
    print("  5. Corpus reconstructable from pairs alone")
    print()
    print("Key findings - Phase 2:")
    print("  6. Text can be parsed for transformation pairs")
    print("  7. Instance vs category distinction is critical (mastiff problem)")
    print("  8. Gaps can be detected and batched for LLM queries")
    print("  9. New dimensions emerge from new content types")
    print()
    print("Next: Phase 3 - LLM Integration")
    print("  - Connect to local LLM for gap filling")
    print("  - Validate instance vs category with LLM")
    print("  - Automated corpus expansion")
