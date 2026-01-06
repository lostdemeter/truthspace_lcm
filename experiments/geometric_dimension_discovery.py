"""
Geometric Dimension Discovery

Replaces statistical co-occurrence with geometric principles:

1. φ-Zipf Weighting (Design 039):
   - φ^(-log(1+freq)) instead of 1/log(1+freq)
   - Same ranking, but derived from geometry
   - Rarity = importance (inward navigation)

2. Tachyon Hypothesis Navigation (Design 053):
   - Navigate backward from hypothesis to evidence
   - "If gender dimension exists, what words support it?"
   - Failed hypotheses are informative

The key insight: We're not discovering dimensions statistically.
We're navigating to dimensions that already exist in the space.

Author: Lesley Gushurst
License: GPLv3
"""

import re
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
import urllib.request
import ssl

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
    
    This is the "inward" direction of φ navigation:
    - φ^n for encoding (outward expansion)
    - φ^(-n) for weighting (inward contraction)
    """
    
    def __init__(self):
        self.word_frequencies: Dict[str, int] = Counter()
        self._cache: Dict[str, float] = {}
    
    def update_frequencies(self, tokens: List[str]):
        """Update word frequency counts from tokens."""
        self.word_frequencies.update(tokens)
        self._cache.clear()
    
    def phi_weight(self, word: str) -> float:
        """
        Geometric weight: φ^(-log(1+freq))
        
        This is equivalent to Zipf for ranking but derived from φ.
        Rare words (low freq) get HIGH weight.
        Common words (high freq) get LOW weight.
        """
        if word in self._cache:
            return self._cache[word]
        
        freq = self.word_frequencies.get(word, 0)
        # φ^(-log(1+freq)) = (1+freq)^(-ln(φ)) ≈ (1+freq)^(-0.481)
        weight = PHI ** (-np.log(1 + freq))
        self._cache[word] = weight
        return weight
    
    def importance(self, word1: str, word2: str) -> float:
        """
        Geometric importance of a word pair.
        
        High importance = both words are rare (meaningful).
        Low importance = one or both words are common (noise).
        """
        return self.phi_weight(word1) * self.phi_weight(word2)
    
    def rank_words(self, words: List[str]) -> List[Tuple[str, float]]:
        """Rank words by geometric importance (highest first)."""
        ranked = [(w, self.phi_weight(w)) for w in set(words)]
        ranked.sort(key=lambda x: -x[1])
        return ranked


# =============================================================================
# TACHYON HYPOTHESIS NAVIGATION (Design 053)
# =============================================================================

@dataclass
class DimensionHypothesis:
    """
    A hypothesis about a dimension that exists in the space.
    
    We navigate BACKWARD from the hypothesis to find evidence.
    """
    name: str
    positive_anchors: List[str]  # Words that should be +1 on this dimension
    negative_anchors: List[str]  # Words that should be -1 on this dimension
    confidence: float = 0.0
    evidence_found: List[str] = field(default_factory=list)
    evidence_missing: List[str] = field(default_factory=list)


class TachyonNavigator:
    """
    Navigate backward from hypotheses to evidence.
    
    Instead of discovering dimensions from co-occurrence (forward),
    we hypothesize dimensions and navigate to find evidence (backward).
    
    This is the Tachyon direction: effect → cause.
    """
    
    def __init__(self, phi_weighting: PhiZipfWeighting):
        self.phi = phi_weighting
        self.hypotheses: List[DimensionHypothesis] = []
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
    
    def hypothesize(self, name: str, 
                    positive: List[str], 
                    negative: List[str]) -> DimensionHypothesis:
        """
        Create a hypothesis about a dimension.
        
        Args:
            name: Dimension name (e.g., "gender")
            positive: Words expected at +1 (e.g., ["he", "king", "man"])
            negative: Words expected at -1 (e.g., ["she", "queen", "woman"])
        """
        hypothesis = DimensionHypothesis(
            name=name,
            positive_anchors=positive,
            negative_anchors=negative
        )
        self.hypotheses.append(hypothesis)
        return hypothesis
    
    def navigate_to_hypothesis(self, hypothesis: DimensionHypothesis) -> float:
        """
        Navigate backward from hypothesis to evidence.
        
        Returns confidence score based on how much evidence we find.
        """
        evidence_found = []
        evidence_missing = []
        total_weight = 0.0
        found_weight = 0.0
        
        # Check positive anchors
        for word in hypothesis.positive_anchors:
            weight = self.phi.phi_weight(word)
            total_weight += weight
            
            if word in self.word_contexts:
                evidence_found.append(word)
                found_weight += weight
            else:
                evidence_missing.append(word)
        
        # Check negative anchors
        for word in hypothesis.negative_anchors:
            weight = self.phi.phi_weight(word)
            total_weight += weight
            
            if word in self.word_contexts:
                evidence_found.append(word)
                found_weight += weight
            else:
                evidence_missing.append(word)
        
        # Confidence = weighted proportion of evidence found
        confidence = found_weight / total_weight if total_weight > 0 else 0.0
        
        hypothesis.confidence = confidence
        hypothesis.evidence_found = evidence_found
        hypothesis.evidence_missing = evidence_missing
        
        return confidence
    
    def discover_contrasts(self, hypothesis: DimensionHypothesis) -> List[Tuple[str, str, float]]:
        """
        Discover contrasting pairs that support the hypothesis.
        
        For each positive anchor found, find words that:
        1. Share similar contexts (substitutable)
        2. Are NOT in the same sentences (contrasting)
        3. Have high φ-weight (meaningful)
        """
        contrasts = []
        
        for pos_word in hypothesis.evidence_found:
            if pos_word not in hypothesis.positive_anchors:
                continue
            
            pos_context = self.word_contexts.get(pos_word, set())
            
            for neg_word in hypothesis.evidence_found:
                if neg_word not in hypothesis.negative_anchors:
                    continue
                
                neg_context = self.word_contexts.get(neg_word, set())
                
                # Context similarity (Jaccard)
                intersection = len(pos_context & neg_context)
                union = len(pos_context | neg_context)
                similarity = intersection / union if union > 0 else 0
                
                # Weight by φ-importance
                importance = self.phi.importance(pos_word, neg_word)
                
                # Score = similarity × importance
                score = similarity * importance
                
                if score > 0:
                    contrasts.append((pos_word, neg_word, score))
        
        contrasts.sort(key=lambda x: -x[2])
        return contrasts
    
    def expand_dimension(self, hypothesis: DimensionHypothesis) -> Dict[str, float]:
        """
        Expand a dimension by finding more words that fit.
        
        Navigate from known anchors to discover new words
        that belong on this dimension.
        """
        expanded = {}
        
        # Start with known anchors
        for word in hypothesis.positive_anchors:
            if word in self.word_contexts:
                expanded[word] = 1.0
        
        for word in hypothesis.negative_anchors:
            if word in self.word_contexts:
                expanded[word] = -1.0
        
        # Find words that co-occur with positive but not negative (and vice versa)
        pos_contexts = set()
        neg_contexts = set()
        
        for word in hypothesis.positive_anchors:
            if word in self.word_contexts:
                pos_contexts.update(self.word_contexts[word])
        
        for word in hypothesis.negative_anchors:
            if word in self.word_contexts:
                neg_contexts.update(self.word_contexts[word])
        
        # Words that appear with positive but not negative → likely positive
        pos_only = pos_contexts - neg_contexts - set(expanded.keys())
        for word in pos_only:
            weight = self.phi.phi_weight(word)
            if weight > 0.1:  # Only meaningful words
                expanded[word] = 0.5  # Tentative positive
        
        # Words that appear with negative but not positive → likely negative
        neg_only = neg_contexts - pos_contexts - set(expanded.keys())
        for word in neg_only:
            weight = self.phi.phi_weight(word)
            if weight > 0.1:
                expanded[word] = -0.5  # Tentative negative
        
        return expanded


# =============================================================================
# GEOMETRIC DIMENSION REGISTRY
# =============================================================================

class GeometricDimensionRegistry:
    """
    Registry that discovers dimensions geometrically.
    
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
        self._next_index = 0
    
    def ingest_text(self, text: str):
        """Ingest text and build geometric structures."""
        tokens = self._tokenize(text)
        self.phi_weighting.update_frequencies(tokens)
        self.navigator.build_contexts(tokens)
        print(f"Ingested {len(tokens)} tokens, {len(set(tokens))} unique")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b[a-z]+\b', text.lower())
    
    def hypothesize_dimension(self, name: str,
                               positive: List[str],
                               negative: List[str]) -> DimensionHypothesis:
        """
        Hypothesize a dimension and navigate to find evidence.
        
        Returns the hypothesis with confidence and evidence.
        """
        hypothesis = self.navigator.hypothesize(name, positive, negative)
        confidence = self.navigator.navigate_to_hypothesis(hypothesis)
        
        print(f"Hypothesis '{name}': confidence={confidence:.3f}")
        print(f"  Found: {hypothesis.evidence_found}")
        print(f"  Missing: {hypothesis.evidence_missing}")
        
        return hypothesis
    
    def register_dimension(self, hypothesis: DimensionHypothesis,
                           min_confidence: float = 0.3) -> bool:
        """
        Register a dimension if hypothesis has sufficient confidence.
        """
        if hypothesis.confidence < min_confidence:
            print(f"  Rejected: confidence {hypothesis.confidence:.3f} < {min_confidence}")
            return False
        
        if hypothesis.name in self._dimensions:
            print(f"  Already registered: {hypothesis.name}")
            return False
        
        # Expand dimension to find more words
        expanded = self.navigator.expand_dimension(hypothesis)
        
        # Register
        idx = self._next_index
        self._dimensions[hypothesis.name] = idx
        self._index_to_name[idx] = hypothesis.name
        self._anchors[hypothesis.name] = expanded
        self._next_index += 1
        
        print(f"  Registered dimension [{idx}] '{hypothesis.name}' with {len(expanded)} anchors")
        return True
    
    # Common function words to exclude from proper noun detection
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
    }
    
    def discover_proper_nouns(self) -> List[Tuple[str, float, float]]:
        """
        Discover proper nouns using Tachyon navigation.
        
        Proper nouns have a unique signature:
        1. They DON'T fit semantic dimensions (not adjectives/adverbs)
        2. They appear with MANY different context words (high connectivity)
        3. They have moderate-to-high frequency (characters appear often)
        4. Their contexts include dimension words but they themselves aren't
        
        The Tachyon insight: Proper nouns are ENTITIES that dimensions
        describe, not the dimensions themselves.
        """
        proper_nouns = []
        
        all_words = list(self.navigator.word_contexts.keys())
        
        for word in all_words:
            # Skip very short words
            if len(word) < 4:
                continue
            
            # Skip function words
            if word in self.FUNCTION_WORDS:
                continue
            
            # Skip if it fits a known dimension (it's a descriptor, not entity)
            fits_dimension = False
            for dim_name, anchors in self._anchors.items():
                if word in anchors:
                    fits_dimension = True
                    break
            
            if fits_dimension:
                continue
            
            context = self.navigator.word_contexts.get(word, set())
            if len(context) < 5:
                continue
            
            # Count how many dimension words appear in this word's context
            dimension_words_in_context = 0
            for ctx_word in context:
                for dim_name, anchors in self._anchors.items():
                    if ctx_word in anchors:
                        dimension_words_in_context += 1
                        break
            
            # Proper noun signature:
            # - High connectivity (appears with many words)
            # - Many dimension words in context (is described by dimensions)
            # - Itself is NOT a dimension word
            connectivity = len(context)
            dimension_density = dimension_words_in_context / len(context) if context else 0
            
            # Score: connectivity × dimension_density
            # High score = entity that is described by many dimensions
            score = connectivity * dimension_density
            
            freq = self.phi_weighting.word_frequencies.get(word, 0)
            
            # Filter: must have reasonable frequency and dimension density
            if freq >= 5 and dimension_density > 0.1:
                proper_nouns.append((word, score, dimension_density, freq))
        
        # Sort by score (highest first)
        proper_nouns.sort(key=lambda x: -x[1])
        
        # Return top results with (word, score, dimension_density)
        return [(w, s, d) for w, s, d, f in proper_nouns[:50]]
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to dimension vector."""
        tokens = self._tokenize(text)
        vector = np.zeros(self._next_index)
        
        for token in tokens:
            for dim_name, anchors in self._anchors.items():
                if token in anchors:
                    idx = self._dimensions[dim_name]
                    level = anchors[token]
                    # Use max aggregation
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
    
    @property
    def num_dims(self) -> int:
        return self._next_index
    
    def summary(self) -> str:
        """Get summary of registered dimensions."""
        lines = [f"Registered {self.num_dims} dimensions:"]
        for dim_name, idx in sorted(self._dimensions.items(), key=lambda x: x[1]):
            anchors = self._anchors.get(dim_name, {})
            pos = [w for w, v in anchors.items() if v > 0][:5]
            neg = [w for w, v in anchors.items() if v < 0][:5]
            lines.append(f"  [{idx:2d}] {dim_name}: {pos} ↔ {neg}")
        return "\n".join(lines)


# =============================================================================
# CORPUS LOADING
# =============================================================================

def fetch_gutenberg(book_id: int) -> str:
    """Fetch a book from Project Gutenberg."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    
    try:
        with urllib.request.urlopen(url, timeout=30, context=context) as response:
            text = response.read().decode('utf-8', errors='ignore')
            start = text.find("*** START OF")
            end = text.find("*** END OF")
            if start != -1 and end != -1:
                start = text.find("\n", start) + 1
                text = text[start:end]
            return text
    except Exception as e:
        raise ValueError(f"Could not fetch book {book_id}: {e}")


def load_sample_text() -> str:
    """Sample text with clear dimensional contrasts."""
    return """
    The king sat upon his golden throne, surveying his vast kingdom.
    The queen stood beside him, her crown glittering with jewels.
    He spoke loudly, his voice echoing through the great hall.
    She whispered softly, her words meant only for him.
    
    The prince rode swiftly through the forest on his white stallion.
    The princess walked slowly through the garden, admiring the roses.
    He was brave and bold, fearing nothing in his path.
    She was gentle and kind, beloved by all who knew her.
    
    The old man shuffled down the dusty road, his back bent with age.
    The young boy ran ahead, full of energy and excitement.
    The ancient castle loomed on the hilltop, its towers crumbling.
    The modern city sprawled below, its lights twinkling in the night.
    
    The rich lord lived in a grand palace with many servants.
    The poor peasant dwelt in a humble cottage with nothing.
    The master commanded with authority and power.
    The servant obeyed quietly, head bowed in submission.
    
    The brave knight charged into battle, sword raised high.
    The cowardly squire fled in terror, dropping his shield.
    The wise sage spoke in riddles, his meaning obscure.
    The foolish jester laughed and tumbled, caring nothing for wisdom.
    
    In the bright morning light, the world seemed hopeful.
    In the dark evening shadows, fears awakened.
    The hot summer sun blazed upon the parched earth.
    The cold winter snow fell upon the frozen ground.
    """


# =============================================================================
# DEMONSTRATIONS
# =============================================================================

def demo_phi_zipf():
    """Demonstrate φ-Zipf weighting."""
    print("=" * 70)
    print("φ-ZIPF WEIGHTING (Design 039)")
    print("=" * 70)
    print()
    
    text = load_sample_text()
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    
    phi = PhiZipfWeighting()
    phi.update_frequencies(tokens)
    
    # Show weights for various words
    test_words = ['the', 'and', 'his', 'king', 'queen', 'throne', 'palace', 'brave', 'cowardly']
    
    print("Word weights (φ^(-log(1+freq))):")
    print("-" * 40)
    for word in test_words:
        freq = phi.word_frequencies.get(word, 0)
        weight = phi.phi_weight(word)
        print(f"  {word:12s} freq={freq:3d}  weight={weight:.4f}")
    
    print()
    print("Top 10 words by φ-weight (most meaningful):")
    ranked = phi.rank_words(tokens)
    for word, weight in ranked[:10]:
        freq = phi.word_frequencies.get(word, 0)
        print(f"  {word:12s} freq={freq:3d}  weight={weight:.4f}")
    
    print()
    print("KEY INSIGHT:")
    print("  Rare words (throne, palace) have HIGH weight.")
    print("  Common words (the, and) have LOW weight.")
    print("  This is Zipf weighting derived from φ geometry!")


def demo_tachyon_navigation():
    """Demonstrate Tachyon hypothesis navigation."""
    print("=" * 70)
    print("TACHYON HYPOTHESIS NAVIGATION (Design 053)")
    print("=" * 70)
    print()
    
    text = load_sample_text()
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    
    phi = PhiZipfWeighting()
    phi.update_frequencies(tokens)
    
    navigator = TachyonNavigator(phi)
    navigator.build_contexts(tokens)
    
    # Hypothesize dimensions
    print("Navigating backward from hypotheses to evidence...")
    print()
    
    hypotheses = [
        ("gender", ["he", "him", "his", "king", "prince", "man", "boy"],
                   ["she", "her", "queen", "princess", "woman", "girl"]),
        ("age", ["old", "ancient", "aged", "elderly"],
               ["young", "youthful", "child", "boy", "girl"]),
        ("courage", ["brave", "bold", "fearless", "knight"],
                    ["cowardly", "timid", "afraid", "fled"]),
        ("wealth", ["rich", "palace", "gold", "grand"],
                   ["poor", "humble", "cottage", "nothing"]),
        ("volume", ["loudly", "shouted", "echoing"],
                   ["softly", "whispered", "quietly"]),
        ("nonexistent", ["xyzzy", "plugh", "frobnitz"],
                        ["quux", "baz", "corge"]),
    ]
    
    for name, positive, negative in hypotheses:
        h = navigator.hypothesize(name, positive, negative)
        confidence = navigator.navigate_to_hypothesis(h)
        
        print(f"Hypothesis: {name}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Evidence found: {h.evidence_found}")
        print(f"  Evidence missing: {h.evidence_missing}")
        
        if confidence > 0.3:
            contrasts = navigator.discover_contrasts(h)
            if contrasts:
                print(f"  Contrasting pairs:")
                for pos, neg, score in contrasts[:3]:
                    print(f"    {pos} ↔ {neg} (score={score:.4f})")
        print()
    
    print("KEY INSIGHT:")
    print("  We navigate BACKWARD from hypothesis to evidence.")
    print("  High confidence = dimension exists in the data.")
    print("  Low confidence = dimension not supported (like 'nonexistent').")
    print("  Failed hypotheses are informative!")


def demo_geometric_discovery():
    """Demonstrate full geometric dimension discovery."""
    print("=" * 70)
    print("GEOMETRIC DIMENSION DISCOVERY")
    print("=" * 70)
    print()
    
    registry = GeometricDimensionRegistry(max_dims=64)
    
    # Ingest text
    text = load_sample_text()
    registry.ingest_text(text)
    print()
    
    # Hypothesize and register dimensions
    dimension_hypotheses = [
        ("gender", ["he", "him", "his", "king", "prince"],
                   ["she", "her", "queen", "princess"]),
        ("age", ["old", "ancient", "aged"],
               ["young", "youthful", "boy", "girl"]),
        ("courage", ["brave", "bold", "knight"],
                    ["cowardly", "timid", "fled"]),
        ("wealth", ["rich", "palace", "grand"],
                   ["poor", "humble", "cottage"]),
        ("volume", ["loudly", "shouted"],
                   ["softly", "whispered", "quietly"]),
        ("speed", ["swiftly", "ran", "charged"],
                  ["slowly", "walked", "shuffled"]),
        ("light", ["bright", "morning", "light"],
                  ["dark", "evening", "shadows"]),
        ("temperature", ["hot", "summer", "blazed"],
                        ["cold", "winter", "frozen"]),
    ]
    
    print("Hypothesizing dimensions...")
    print("-" * 40)
    
    for name, positive, negative in dimension_hypotheses:
        h = registry.hypothesize_dimension(name, positive, negative)
        registry.register_dimension(h, min_confidence=0.2)
        print()
    
    print(registry.summary())
    print()
    
    # Discover proper nouns
    print("Discovering proper nouns (Tachyon navigation)...")
    print("-" * 40)
    proper_nouns = registry.discover_proper_nouns()
    
    if proper_nouns:
        print("Potential proper nouns (high self-weight, low context weight):")
        for word, weight, ctx_weight in proper_nouns[:10]:
            print(f"  {word:15s} self={weight:.3f} context={ctx_weight:.3f}")
    else:
        print("  No proper nouns detected (sample text has none)")
    
    print()
    
    # Test encoding
    print("ENCODING TEST:")
    print("-" * 40)
    
    test_sentences = [
        "The brave king spoke loudly",
        "The timid queen whispered softly",
        "The old man walked slowly in the cold",
        "The young boy ran swiftly in the heat",
    ]
    
    for sentence in test_sentences:
        vector = registry.encode_text(sentence)
        desc = registry.describe_vector(vector)
        print(f"  '{sentence}'")
        print(f"    → {desc}")
        print()


def demo_gutenberg():
    """Demonstrate on Project Gutenberg text."""
    print("=" * 70)
    print("GEOMETRIC DISCOVERY - PRIDE AND PREJUDICE")
    print("=" * 70)
    print()
    
    print("Fetching Pride and Prejudice...")
    try:
        text = fetch_gutenberg(1342)
        print(f"Fetched {len(text)} characters")
    except Exception as e:
        print(f"Could not fetch: {e}")
        print("Using sample text instead...")
        text = load_sample_text() * 20
    
    print()
    
    registry = GeometricDimensionRegistry(max_dims=64)
    registry.ingest_text(text)
    print()
    
    # Hypothesize dimensions relevant to Pride and Prejudice
    dimension_hypotheses = [
        ("gender", ["he", "him", "his", "mr", "gentleman", "man"],
                   ["she", "her", "mrs", "miss", "lady", "woman"]),
        ("social_class", ["rich", "wealthy", "estate", "fortune", "noble"],
                         ["poor", "humble", "servant", "common"]),
        ("emotion", ["happy", "pleased", "delighted", "joy"],
                    ["unhappy", "distressed", "mortified", "shame"]),
        ("approval", ["good", "amiable", "agreeable", "pleasant"],
                     ["bad", "disagreeable", "proud", "arrogant"]),
    ]
    
    print("Hypothesizing dimensions...")
    print("-" * 40)
    
    for name, positive, negative in dimension_hypotheses:
        h = registry.hypothesize_dimension(name, positive, negative)
        registry.register_dimension(h, min_confidence=0.3)
        print()
    
    print(registry.summary())
    print()
    
    # Discover proper nouns
    print("Discovering proper nouns...")
    print("-" * 40)
    proper_nouns = registry.discover_proper_nouns()
    
    print("Top potential proper nouns (entities described by dimensions):")
    for word, score, dim_density in proper_nouns[:15]:
        print(f"  {word:15s} score={score:.1f} dim_density={dim_density:.2f}")
    
    print()
    print("KEY INSIGHT:")
    print("  Proper nouns (elizabeth, darcy, bennet, bingley, jane, lydia...)")
    print("  are ENTITIES that dimensions DESCRIBE.")
    print("  They have high connectivity + high dimension density.")
    print("  They don't fit dimensions themselves - they ARE the subjects.")
    print()
    print("  This is Tachyon navigation: we hypothesized 'what are entities?'")
    print("  and navigated backward to find evidence (dimension words in context).")


if __name__ == "__main__":
    demo_phi_zipf()
    print("\n" + "=" * 70 + "\n")
    demo_tachyon_navigation()
    print("\n" + "=" * 70 + "\n")
    demo_geometric_discovery()
    print("\n" + "=" * 70 + "\n")
    demo_gutenberg()
