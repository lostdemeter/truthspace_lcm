"""
Dynamic Dimension Discovery and Registry

Implements geometric dimension discovery using:
1. φ-Zipf Weighting (Design 039): φ^(-log(1+freq)) for importance
2. Tachyon Hypothesis Navigation (Design 053): Navigate backward to evidence
3. Proper Noun Discovery: Entities described by dimensions

This replaces statistical co-occurrence with geometric principles.
The key insight: We're not discovering dimensions statistically.
We're navigating to dimensions that already exist in the space.

Author: Lesley Gushurst
License: GPLv3
"""

import re
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# φ-ZIPF WEIGHTING (Design 039)
# =============================================================================

class PhiZipfWeighting:
    """
    Geometric weighting using φ powers instead of statistical Zipf.
    
    φ^(-log(1+freq)) produces identical rankings to 1/log(1+freq)
    but is derived from geometry, not statistics.
    """
    
    def __init__(self):
        self.word_frequencies: Counter = Counter()
        self._cache: Dict[str, float] = {}
    
    def update_frequencies(self, tokens: List[str]):
        """Update word frequency counts from tokens."""
        self.word_frequencies.update(tokens)
        self._cache.clear()
    
    def phi_weight(self, word: str) -> float:
        """
        Geometric weight: φ^(-log(1+freq))
        
        Rare words (low freq) get HIGH weight.
        Common words (high freq) get LOW weight.
        """
        if word in self._cache:
            return self._cache[word]
        
        freq = self.word_frequencies.get(word, 0)
        weight = PHI ** (-np.log(1 + freq))
        self._cache[word] = weight
        return weight
    
    def importance(self, word1: str, word2: str) -> float:
        """Geometric importance of a word pair."""
        return self.phi_weight(word1) * self.phi_weight(word2)


# =============================================================================
# TACHYON HYPOTHESIS NAVIGATION (Design 053)
# =============================================================================

@dataclass
class DimensionHypothesis:
    """A hypothesis about a dimension that exists in the space."""
    name: str
    positive_anchors: List[str]
    negative_anchors: List[str]
    confidence: float = 0.0
    evidence_found: List[str] = field(default_factory=list)
    evidence_missing: List[str] = field(default_factory=list)


class TachyonNavigator:
    """
    Navigate backward from hypotheses to evidence.
    
    Instead of discovering dimensions from co-occurrence (forward),
    we hypothesize dimensions and navigate to find evidence (backward).
    """
    
    def __init__(self, phi_weighting: PhiZipfWeighting):
        self.phi = phi_weighting
        self.word_contexts: Dict[str, Set[str]] = {}
    
    def build_contexts(self, tokens: List[str], window_size: int = 5):
        """Build context windows for each word."""
        for i, word in enumerate(tokens):
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            
            context = set(tokens[j] for j in range(start, end) if j != i)
            
            if word not in self.word_contexts:
                self.word_contexts[word] = set()
            self.word_contexts[word].update(context)
    
    def navigate_to_hypothesis(self, hypothesis: DimensionHypothesis) -> float:
        """
        Navigate backward from hypothesis to evidence.
        Returns confidence score based on how much evidence we find.
        """
        evidence_found = []
        evidence_missing = []
        total_weight = 0.0
        found_weight = 0.0
        
        for word in hypothesis.positive_anchors + hypothesis.negative_anchors:
            weight = self.phi.phi_weight(word)
            total_weight += weight
            
            if word in self.word_contexts:
                evidence_found.append(word)
                found_weight += weight
            else:
                evidence_missing.append(word)
        
        confidence = found_weight / total_weight if total_weight > 0 else 0.0
        
        hypothesis.confidence = confidence
        hypothesis.evidence_found = evidence_found
        hypothesis.evidence_missing = evidence_missing
        
        return confidence
    
    def expand_dimension(self, hypothesis: DimensionHypothesis) -> Dict[str, float]:
        """
        Expand a dimension by finding more words that fit.
        Navigate from known anchors to discover new words.
        """
        expanded = {}
        
        for word in hypothesis.positive_anchors:
            if word in self.word_contexts:
                expanded[word] = 1.0
        
        for word in hypothesis.negative_anchors:
            if word in self.word_contexts:
                expanded[word] = -1.0
        
        pos_contexts = set()
        neg_contexts = set()
        
        for word in hypothesis.positive_anchors:
            if word in self.word_contexts:
                pos_contexts.update(self.word_contexts[word])
        
        for word in hypothesis.negative_anchors:
            if word in self.word_contexts:
                neg_contexts.update(self.word_contexts[word])
        
        pos_only = pos_contexts - neg_contexts - set(expanded.keys())
        for word in pos_only:
            weight = self.phi.phi_weight(word)
            if weight > 0.1:
                expanded[word] = 0.5
        
        neg_only = neg_contexts - pos_contexts - set(expanded.keys())
        for word in neg_only:
            weight = self.phi.phi_weight(word)
            if weight > 0.1:
                expanded[word] = -0.5
        
        return expanded


# =============================================================================
# DYNAMIC DIMENSION REGISTRY
# =============================================================================

# Known dimension patterns for bootstrapping
BOOTSTRAP_DIMENSIONS = {
    'gender': {
        'he': 1.0, 'him': 1.0, 'his': 1.0, 'himself': 1.0,
        'she': -1.0, 'her': -1.0, 'hers': -1.0, 'herself': -1.0,
        'man': 1.0, 'woman': -1.0, 'boy': 1.0, 'girl': -1.0,
        'king': 1.0, 'queen': -1.0, 'prince': 1.0, 'princess': -1.0,
        'lord': 1.0, 'lady': -1.0, 'sir': 1.0, 'madam': -1.0,
        'gentleman': 1.0, 'father': 1.0, 'mother': -1.0,
        'son': 1.0, 'daughter': -1.0, 'brother': 1.0, 'sister': -1.0,
        'husband': 1.0, 'wife': -1.0, 'mr': 1.0, 'mrs': -1.0, 'miss': -1.0,
    },
    'regality': {
        'king': 2.0, 'queen': 2.0, 'prince': 1.5, 'princess': 1.5,
        'royal': 2.0, 'regal': 2.0, 'noble': 1.5, 'aristocrat': 1.5,
        'palace': 2.0, 'castle': 1.5, 'throne': 2.0, 'crown': 2.0,
        'finery': 1.5, 'jewels': 1.5, 'silk': 1.0, 'velvet': 1.0,
        'servant': -1.0, 'peasant': -1.5, 'common': -1.0, 'humble': -0.5,
        'cottage': -1.0, 'hovel': -1.5, 'rags': -1.5,
    },
    'age': {
        'old': 1.0, 'elderly': 1.0, 'aged': 1.0, 'ancient': 1.5,
        'young': -1.0, 'youthful': -1.0, 'child': -1.5, 'baby': -2.0,
        'senior': 0.5, 'junior': -0.5,
    },
    'size': {
        'big': 1.0, 'large': 1.0, 'huge': 1.5, 'enormous': 2.0, 'giant': 2.0,
        'small': -1.0, 'little': -1.0, 'tiny': -1.5, 'minute': -2.0,
    },
    'speed': {
        'fast': 1.0, 'quick': 1.0, 'swift': 1.0, 'rapid': 1.0,
        'quickly': 1.0, 'swiftly': 1.0, 'rapidly': 1.0,
        'slow': -1.0, 'sluggish': -1.0, 'leisurely': -0.5,
        'slowly': -1.0, 'gradually': -0.5,
    },
    'volume': {
        'loud': 1.0, 'loudly': 1.0, 'shouted': 1.0, 'roared': 1.5,
        'screamed': 1.5, 'yelled': 1.0,
        'quiet': -1.0, 'quietly': -1.0, 'whispered': -1.0, 'murmured': -1.0,
        'softly': -1.0,
    },
    'temperature': {
        'hot': 1.0, 'warm': 0.5, 'burning': 1.5, 'blazing': 1.5,
        'cold': -1.0, 'cool': -0.5, 'freezing': -1.5, 'frozen': -1.5,
        'summer': 1.0, 'winter': -1.0,
    },
    'light': {
        'bright': 1.0, 'light': 0.5, 'brilliant': 1.5, 'radiant': 1.5,
        'dark': -1.0, 'dim': -0.5, 'shadow': -1.0, 'gloomy': -1.0,
        'day': 0.5, 'night': -0.5, 'morning': 0.5, 'evening': -0.5,
    },
    'wealth': {
        'rich': 1.0, 'wealthy': 1.0, 'prosperous': 1.0,
        'gold': 1.0, 'fortune': 1.0, 'treasure': 1.0,
        'poor': -1.0, 'destitute': -1.5, 'impoverished': -1.0,
    },
    'courage': {
        'brave': 1.0, 'bold': 1.0, 'courageous': 1.0, 'fearless': 1.5,
        'hero': 1.0, 'heroic': 1.0,
        'cowardly': -1.0, 'timid': -0.5, 'afraid': -0.5, 'coward': -1.0,
    },
    'wisdom': {
        'wise': 1.0, 'sage': 1.0, 'clever': 0.5, 'intelligent': 0.5,
        'foolish': -1.0, 'stupid': -1.0, 'ignorant': -1.0, 'fool': -1.0,
    },
    'beauty': {
        'beautiful': 1.0, 'lovely': 1.0, 'handsome': 1.0, 'gorgeous': 1.5,
        'pretty': 0.5, 'elegant': 1.0,
        'ugly': -1.0, 'hideous': -1.5, 'plain': -0.5,
    },
    'good_evil': {
        'good': 1.0, 'kind': 1.0, 'gentle': 0.5, 'virtuous': 1.0,
        'evil': -1.0, 'wicked': -1.0, 'cruel': -1.0, 'malicious': -1.0,
    },
    'formality': {
        'formal': 1.0, 'proper': 1.0, 'dignified': 1.0,
        'informal': -1.0, 'casual': -0.5, 'relaxed': -0.5,
    },
    'certainty': {
        'certain': 1.0, 'sure': 1.0, 'definite': 1.0,
        'uncertain': -1.0, 'unsure': -1.0, 'doubtful': -1.0,
    },
}

# Function words to exclude from entity detection
FUNCTION_WORDS = {
    'that', 'with', 'have', 'which', 'from', 'were', 'they', 'very',
    'could', 'been', 'their', 'this', 'would', 'them', 'what', 'when',
    'there', 'will', 'more', 'some', 'than', 'only', 'other', 'such',
    'into', 'most', 'also', 'made', 'after', 'being', 'much', 'even',
    'before', 'where', 'those', 'these', 'should', 'might', 'must',
    'about', 'over', 'just', 'like', 'back', 'still', 'well', 'here',
    'then', 'first', 'last', 'long', 'great', 'little', 'every', 'never',
    'under', 'always', 'between', 'often', 'however', 'though', 'without',
    'again', 'same', 'another', 'while', 'each', 'both', 'through',
    'during', 'until', 'against', 'among', 'within', 'along', 'since',
    'upon', 'away', 'down', 'ever', 'yet', 'already', 'soon', 'rather',
    'quite', 'almost', 'perhaps', 'indeed', 'certainly', 'probably',
    'nothing', 'something', 'anything', 'everything', 'everyone',
    'anyone', 'someone', 'nobody', 'everybody', 'anybody', 'somebody',
    'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves',
    'themselves', 'whose', 'whom', 'whatever', 'whoever', 'whichever',
    'your', 'said', 'know', 'think', 'make', 'time', 'take', 'come',
}


class DynamicDimensionRegistry:
    """
    Registry that discovers and manages dimensions geometrically.
    
    Uses φ-Zipf weighting for importance and
    Tachyon navigation for hypothesis-driven discovery.
    """
    
    def __init__(self, max_dims: int = 128):
        self.max_dims = max_dims
        self.phi_weighting = PhiZipfWeighting()
        self.navigator = TachyonNavigator(self.phi_weighting)
        
        self._dimensions: Dict[str, int] = {}
        self._index_to_name: Dict[int, str] = {}
        self._anchors: Dict[str, Dict[str, float]] = {}
        self._entities: Dict[str, Dict[str, Any]] = {}  # Discovered entities
        self._next_index = 0
        
        # Bootstrap with known dimensions
        self._bootstrap()
    
    def _bootstrap(self):
        """Bootstrap with known dimension patterns."""
        for dim_name, anchors in BOOTSTRAP_DIMENSIONS.items():
            if self._next_index >= self.max_dims:
                break
            
            idx = self._next_index
            self._dimensions[dim_name] = idx
            self._index_to_name[idx] = dim_name
            self._anchors[dim_name] = anchors.copy()
            self._next_index += 1
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b[a-z]+\b', text.lower())
    
    def ingest_text(self, text: str):
        """Ingest text and build geometric structures."""
        tokens = self.tokenize(text)
        self.phi_weighting.update_frequencies(tokens)
        self.navigator.build_contexts(tokens)
    
    def hypothesize_dimension(self, name: str,
                               positive: List[str],
                               negative: List[str],
                               min_confidence: float = 0.3) -> Optional[DimensionHypothesis]:
        """
        Hypothesize a dimension and register if confident.
        """
        hypothesis = DimensionHypothesis(
            name=name,
            positive_anchors=positive,
            negative_anchors=negative
        )
        
        confidence = self.navigator.navigate_to_hypothesis(hypothesis)
        
        if confidence >= min_confidence and name not in self._dimensions:
            expanded = self.navigator.expand_dimension(hypothesis)
            
            idx = self._next_index
            self._dimensions[name] = idx
            self._index_to_name[idx] = name
            self._anchors[name] = expanded
            self._next_index += 1
            
            return hypothesis
        
        return None
    
    def discover_entities(self) -> List[Tuple[str, float, float]]:
        """
        Discover entities (proper nouns) using Tachyon navigation.
        
        Entities are described BY dimensions, not dimensions themselves.
        """
        entities = []
        
        for word in self.navigator.word_contexts.keys():
            if len(word) < 4:
                continue
            
            if word in FUNCTION_WORDS:
                continue
            
            # Skip if it fits a known dimension
            fits_dimension = False
            for anchors in self._anchors.values():
                if word in anchors:
                    fits_dimension = True
                    break
            
            if fits_dimension:
                continue
            
            context = self.navigator.word_contexts.get(word, set())
            if len(context) < 5:
                continue
            
            # Count dimension words in context
            dimension_words_in_context = 0
            for ctx_word in context:
                for anchors in self._anchors.values():
                    if ctx_word in anchors:
                        dimension_words_in_context += 1
                        break
            
            connectivity = len(context)
            dimension_density = dimension_words_in_context / len(context) if context else 0
            score = connectivity * dimension_density
            
            freq = self.phi_weighting.word_frequencies.get(word, 0)
            
            if freq >= 5 and dimension_density > 0.1:
                entities.append((word, score, dimension_density))
                self._entities[word] = {
                    'score': score,
                    'dimension_density': dimension_density,
                    'frequency': freq,
                }
        
        entities.sort(key=lambda x: -x[1])
        return entities[:50]
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to dimension vector using MAX aggregation."""
        tokens = self.tokenize(text)
        vector = np.zeros(self._next_index)
        
        for token in tokens:
            for dim_name, anchors in self._anchors.items():
                if token in anchors:
                    idx = self._dimensions[dim_name]
                    level = anchors[token]
                    if abs(level) > abs(vector[idx]):
                        vector[idx] = level
        
        return vector
    
    def describe_vector(self, vector: np.ndarray) -> Dict[str, float]:
        """Get human-readable description of a dimension vector."""
        desc = {}
        for i, val in enumerate(vector):
            if abs(val) > 0.01:
                name = self._index_to_name.get(i, f"dim_{i}")
                desc[name] = float(val)
        return desc
    
    def get_word_dimensions(self, word: str) -> Dict[str, float]:
        """Get all dimension activations for a word."""
        word = word.lower()
        activations = {}
        
        for dim_name, anchors in self._anchors.items():
            if word in anchors:
                activations[dim_name] = anchors[word]
        
        return activations
    
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Cosine similarity between two dimension vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Euclidean distance between two dimension vectors."""
        return float(np.linalg.norm(vec1 - vec2))
    
    @property
    def num_dims(self) -> int:
        return self._next_index
    
    @property
    def dimension_names(self) -> List[str]:
        return list(self._dimensions.keys())
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of registered dimensions."""
        return {
            'num_dimensions': self._next_index,
            'dimensions': list(self._dimensions.keys()),
            'num_entities': len(self._entities),
            'top_entities': list(self._entities.keys())[:10],
        }
    
    def __repr__(self) -> str:
        return f"DynamicDimensionRegistry({self._next_index} dims, {len(self._entities)} entities)"
