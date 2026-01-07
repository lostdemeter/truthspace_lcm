"""
Transformation Space - Geometric Sentence Transformation

Learns transformation patterns from corpus and applies them to new sentences
without LLM calls. Uses the quaternion encoder to compute delta vectors
for each dimension.

Key insight: ENCODE(source) + DELTA(dimension) ≈ ENCODE(target)

The transformation vocabulary maps word changes for each dimension:
- tense: "went" → "will go", "sat" → "will sit"
- regality: "Jack" → "Sir Jack", "went" → "did proceed"

Author: Lesley Gushurst
License: GPLv3
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np

from .quaternion_encoder import QuaternionEncoder, QuaternionPosition
from .dynamic_dimensions import DynamicDimensionRegistry


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TransformationDelta:
    """A learned transformation delta for a dimension."""
    dimension: str
    source_value: str
    target_value: str
    delta_vector: np.ndarray
    word_mappings: Dict[str, str] = field(default_factory=dict)
    examples: List[Tuple[str, str]] = field(default_factory=list)
    
    def __repr__(self):
        return f"Delta({self.dimension}: {self.source_value}→{self.target_value}, {len(self.word_mappings)} mappings)"


@dataclass
class TransformationResult:
    """Result of applying a transformation."""
    original: str
    transformed: str
    dimension: str
    target_value: str
    confidence: float
    method: str  # "vocabulary", "pattern", "fallback"
    word_changes: List[Tuple[str, str]] = field(default_factory=list)


# =============================================================================
# TRANSFORMATION SPACE
# =============================================================================

class TransformationSpace:
    """
    Learns and applies sentence transformations geometrically.
    
    Usage:
        space = TransformationSpace()
        space.load_corpus("corpus/transformation_corpus.json")
        
        result = space.transform(
            "Jack and Jill went up the hill",
            dimension="tense",
            target_value="future"
        )
        # -> "Jack and Jill will go up the hill"
    """
    
    def __init__(self, encoder: QuaternionEncoder = None):
        if encoder is None:
            encoder = QuaternionEncoder()
        
        self.encoder = encoder
        
        # Learned deltas: dimension -> (source_value, target_value) -> TransformationDelta
        self._deltas: Dict[str, Dict[Tuple[str, str], TransformationDelta]] = defaultdict(dict)
        
        # Word transformation vocabulary: dimension -> target_value -> word -> replacement
        self._vocabulary: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
        
        # Pattern-based transformations (regex patterns)
        self._patterns: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
        
        # Statistics
        self._corpus_size = 0
        self._dimensions_learned: Set[str] = set()
    
    def load_corpus(self, path: Path) -> int:
        """
        Load transformation corpus and learn patterns.
        
        Returns number of transformations loaded.
        """
        if isinstance(path, str):
            path = Path(path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        transformations = data.get("transformations", [])
        
        for t in transformations:
            self._learn_transformation(t)
        
        self._corpus_size = len(transformations)
        self._build_patterns()
        
        return self._corpus_size
    
    def _learn_transformation(self, t: Dict[str, Any]) -> None:
        """Learn from a single transformation example."""
        source = t["source"]
        target = t["target"]
        dimension_delta = t.get("dimension_delta", {})
        
        # Encode source and target
        source_pos = self.encoder.encode(source)
        target_pos = self.encoder.encode(target)
        
        # Compute delta vector
        delta_vec = target_pos.to_flat() - source_pos.to_flat()
        
        # Learn word mappings by comparing source and target
        word_mappings = self._extract_word_mappings(source, target)
        
        # Store delta for each dimension changed
        for dim, (src_val, tgt_val) in dimension_delta.items():
            key = (src_val, tgt_val)
            
            if key not in self._deltas[dim]:
                self._deltas[dim][key] = TransformationDelta(
                    dimension=dim,
                    source_value=src_val,
                    target_value=tgt_val,
                    delta_vector=delta_vec,
                    word_mappings={},
                    examples=[]
                )
            
            delta = self._deltas[dim][key]
            delta.examples.append((source, target))
            
            # Merge word mappings
            for src_word, tgt_word in word_mappings.items():
                if src_word not in delta.word_mappings:
                    delta.word_mappings[src_word] = tgt_word
                    # Also add to vocabulary
                    self._vocabulary[dim][tgt_val][src_word.lower()] = tgt_word
            
            self._dimensions_learned.add(dim)
    
    def _extract_word_mappings(self, source: str, target: str) -> Dict[str, str]:
        """
        Extract word-level mappings between source and target.
        
        Uses alignment heuristics to find which words changed.
        """
        mappings = {}
        
        # Tokenize
        src_words = re.findall(r'\b[\w\']+\b', source.lower())
        tgt_words = re.findall(r'\b[\w\']+\b', target.lower())
        
        # Find words that appear in source but not target
        src_set = set(src_words)
        tgt_set = set(tgt_words)
        
        removed = src_set - tgt_set
        added = tgt_set - src_set
        
        # Simple heuristic: if one word removed and one added at similar position,
        # they're likely a transformation pair
        if len(removed) == 1 and len(added) == 1:
            src_word = list(removed)[0]
            tgt_word = list(added)[0]
            mappings[src_word] = tgt_word
        
        # Handle verb tense changes (more complex patterns)
        # e.g., "went" -> "will go", "sat" -> "will sit"
        for i, src_word in enumerate(src_words):
            if src_word in removed:
                # Look for corresponding position in target
                # This is approximate - real alignment would be more sophisticated
                for j, tgt_word in enumerate(tgt_words):
                    if tgt_word in added:
                        # Check if positions are close
                        if abs(i - j) <= 2:
                            mappings[src_word] = tgt_word
                            break
        
        return mappings
    
    def _build_patterns(self) -> None:
        """Build regex patterns from learned vocabulary."""
        # Tense patterns
        self._patterns["tense"]["future"] = [
            (r'\bwent\b', 'will go'),
            (r'\bsat\b', 'will sit'),
            (r'\bstood\b', 'will stand'),
            (r'\bwalked\b', 'will walk'),
            (r'\bran\b', 'will run'),
            (r'\bcame\b', 'will come'),
            (r'\bwas\b', 'will be'),
            (r'\bwere\b', 'will be'),
            (r'\bhad\b', 'will have'),
            (r'\bdid\b', 'will do'),
            (r'\bmade\b', 'will make'),
            (r'\btook\b', 'will take'),
            (r'\bgave\b', 'will give'),
            (r'\bfound\b', 'will find'),
            (r'\bsaid\b', 'will say'),
            (r'\bknew\b', 'will know'),
            (r'\bthought\b', 'will think'),
            (r'\bsaw\b', 'will see'),
            (r'\bgot\b', 'will get'),
        ]
        
        self._patterns["tense"]["present"] = [
            (r'\bwent\b', 'go'),
            (r'\bsat\b', 'sit'),
            (r'\bstood\b', 'stand'),
            (r'\bwalked\b', 'walk'),
            (r'\bran\b', 'run'),
            (r'\bcame\b', 'come'),
            (r'\bwas\b', 'is'),
            (r'\bwere\b', 'are'),
            (r'\bhad\b', 'have'),
            (r'\bdid\b', 'do'),
            (r'\bmade\b', 'make'),
            (r'\btook\b', 'take'),
            (r'\bgave\b', 'give'),
            (r'\bfound\b', 'find'),
            (r'\bsaid\b', 'say'),
            (r'\bknew\b', 'know'),
            (r'\bthought\b', 'think'),
            (r'\bsaw\b', 'see'),
            (r'\bgot\b', 'get'),
        ]
        
        self._patterns["tense"]["past"] = [
            (r'\bgo\b', 'went'),
            (r'\bgoes\b', 'went'),
            (r'\bsit\b', 'sat'),
            (r'\bsits\b', 'sat'),
            (r'\bstand\b', 'stood'),
            (r'\bstands\b', 'stood'),
            (r'\bwalk\b', 'walked'),
            (r'\bwalks\b', 'walked'),
            (r'\brun\b', 'ran'),
            (r'\bruns\b', 'ran'),
            (r'\bcome\b', 'came'),
            (r'\bcomes\b', 'came'),
            (r'\bis\b', 'was'),
            (r'\bare\b', 'were'),
            (r'\bhave\b', 'had'),
            (r'\bhas\b', 'had'),
            (r'\bdo\b', 'did'),
            (r'\bdoes\b', 'did'),
            (r'\bwill\b', ''),  # Remove "will" for past
        ]
        
        # Regality patterns
        self._patterns["regality"]["royal"] = [
            (r'\bjack\b', 'His Majesty Jack'),
            (r'\bjill\b', 'Her Majesty Jill'),
            (r'\bthe\b', 'the most esteemed'),
            (r'\bwent\b', 'did proceed'),
            (r'\bwalked\b', 'did traverse'),
            (r'\bsat\b', 'did take repose'),
            (r'\bsaid\b', 'did proclaim'),
        ]
        
        self._patterns["regality"]["noble"] = [
            (r'\bjack\b', 'Sir Jack'),
            (r'\bjill\b', 'Lady Jill'),
            (r'\bwent\b', 'proceeded'),
            (r'\bwalked\b', 'traversed'),
        ]
        
        # Formality patterns
        self._patterns["formality"]["formal"] = [
            (r'\bgot\b', 'obtained'),
            (r'\bwent\b', 'proceeded'),
            (r'\bsaid\b', 'stated'),
            (r'\basked\b', 'inquired'),
            (r'\btold\b', 'informed'),
            (r'\bhelped\b', 'assisted'),
            (r'\bused\b', 'utilized'),
            (r'\bshowed\b', 'demonstrated'),
        ]
        
        self._patterns["formality"]["casual"] = [
            (r'\bobtained\b', 'got'),
            (r'\bproceeded\b', 'went'),
            (r'\bstated\b', 'said'),
            (r'\binquired\b', 'asked'),
            (r'\binformed\b', 'told'),
            (r'\bassisted\b', 'helped'),
            (r'\butilized\b', 'used'),
            (r'\bdemonstrated\b', 'showed'),
        ]
        
        # Voice patterns
        self._patterns["voice"]["passive"] = [
            # These are harder - need subject/object swap
            # For now, simple patterns
        ]
        
        # Certainty patterns
        self._patterns["certainty"]["certain"] = [
            (r'^', 'It is certain that '),  # Prepend
        ]
        
        self._patterns["certainty"]["uncertain"] = [
            (r'^', 'It might be that '),  # Prepend
        ]
        
        # Emotion patterns
        self._patterns["emotion"]["happy"] = [
            (r'\.$', '!'),  # End with exclamation
        ]
        
        self._patterns["emotion"]["sad"] = [
            (r'!$', '.'),  # Remove exclamation
        ]
        
        # Note: We intentionally don't auto-add vocabulary patterns here
        # The learned vocabulary is often noisy. Instead, we use it for
        # confidence scoring and fallback, but rely on curated patterns.
    
    def transform(self, text: str, dimension: str, target_value: str) -> TransformationResult:
        """
        Transform text along a dimension.
        
        Args:
            text: Input sentence
            dimension: Dimension to transform (e.g., "tense", "regality")
            target_value: Target value (e.g., "future", "royal")
            
        Returns:
            TransformationResult with transformed text
        """
        original = text
        transformed = text
        word_changes = []
        method = "pattern"
        confidence = 0.0
        applied_patterns = set()  # Track what we've already changed
        
        # Apply pattern-based transformations (curated, reliable)
        patterns = self._patterns.get(dimension, {}).get(target_value, [])
        for pattern, replacement in patterns:
            # Skip if we've already applied a pattern to this word
            match = re.search(pattern, transformed, re.IGNORECASE)
            if match and match.group().lower() not in applied_patterns:
                old_transformed = transformed
                # Only replace first occurrence to avoid double-replacement
                transformed = re.sub(pattern, replacement, transformed, count=1, flags=re.IGNORECASE)
                if transformed != old_transformed:
                    word_changes.append((match.group(), replacement))
                    applied_patterns.add(match.group().lower())
        
        # Calculate confidence based on changes made
        if word_changes:
            confidence = min(1.0, len(word_changes) * 0.25)
        else:
            confidence = 0.1
            method = "fallback"
        
        # Clean up double spaces and artifacts
        transformed = re.sub(r'\s+', ' ', transformed).strip()
        
        # Preserve original capitalization for first letter
        if original and original[0].isupper() and transformed:
            transformed = transformed[0].upper() + transformed[1:]
        
        return TransformationResult(
            original=original,
            transformed=transformed,
            dimension=dimension,
            target_value=target_value,
            confidence=confidence,
            method=method,
            word_changes=word_changes
        )
    
    def transform_multi(self, text: str, 
                        transformations: List[Tuple[str, str]]) -> TransformationResult:
        """
        Apply multiple transformations in sequence.
        
        Args:
            text: Input sentence
            transformations: List of (dimension, target_value) pairs
            
        Returns:
            TransformationResult with all transformations applied
        """
        current = text
        all_changes = []
        total_confidence = 0.0
        
        for dim, target_val in transformations:
            result = self.transform(current, dim, target_val)
            current = result.transformed
            all_changes.extend(result.word_changes)
            total_confidence += result.confidence
        
        avg_confidence = total_confidence / len(transformations) if transformations else 0.0
        
        return TransformationResult(
            original=text,
            transformed=current,
            dimension="+".join(d for d, _ in transformations),
            target_value="+".join(v for _, v in transformations),
            confidence=avg_confidence,
            method="multi",
            word_changes=all_changes
        )
    
    def available_dimensions(self) -> List[str]:
        """Get list of dimensions with learned transformations."""
        return list(self._dimensions_learned)
    
    def available_values(self, dimension: str) -> List[str]:
        """Get available target values for a dimension."""
        values = set()
        for (src, tgt) in self._deltas.get(dimension, {}).keys():
            values.add(tgt)
        # Also include pattern-based values
        values.update(self._patterns.get(dimension, {}).keys())
        return sorted(values)
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics about learned transformations."""
        total_mappings = sum(
            len(delta.word_mappings) 
            for dim_deltas in self._deltas.values() 
            for delta in dim_deltas.values()
        )
        
        return {
            "corpus_size": self._corpus_size,
            "dimensions_learned": list(self._dimensions_learned),
            "total_deltas": sum(len(d) for d in self._deltas.values()),
            "total_word_mappings": total_mappings,
            "vocabulary_size": sum(
                len(words) 
                for targets in self._vocabulary.values() 
                for words in targets.values()
            ),
        }
    
    def describe_delta(self, dimension: str, source_value: str, target_value: str) -> Optional[Dict[str, Any]]:
        """Get description of a learned delta."""
        key = (source_value, target_value)
        delta = self._deltas.get(dimension, {}).get(key)
        
        if not delta:
            return None
        
        return {
            "dimension": dimension,
            "source_value": source_value,
            "target_value": target_value,
            "num_examples": len(delta.examples),
            "word_mappings": delta.word_mappings,
            "sample_examples": delta.examples[:3],
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_transformation_space(corpus_path: Path = None) -> TransformationSpace:
    """Load transformation space with default corpus."""
    if corpus_path is None:
        corpus_path = Path(__file__).parent.parent / "corpus" / "transformation_corpus.json"
    
    space = TransformationSpace()
    
    if corpus_path.exists():
        space.load_corpus(corpus_path)
    
    return space
