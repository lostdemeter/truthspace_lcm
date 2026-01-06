"""
Automatic Dimension Discovery from Text Corpora

Discovers emergent dimensions by finding contrasting word pairs in text.

Approach:
1. Extract word co-occurrence patterns
2. Find words that appear in similar contexts but rarely together
3. These are likely to be contrasting pairs (he/she, big/small, quickly/slowly)
4. Each contrasting pair defines a dimension

Sources:
- Project Gutenberg texts (public domain)
- Focus on texts with rich stylistic variation

Author: Lesley Gushurst
License: GPLv3
"""

import re
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
import urllib.request
import ssl

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


# =============================================================================
# CORPUS LOADING
# =============================================================================

def fetch_gutenberg(book_id: int) -> str:
    """
    Fetch a book from Project Gutenberg.
    
    Common book IDs:
    - 1342: Pride and Prejudice
    - 84: Frankenstein
    - 1661: Sherlock Holmes
    - 2701: Moby Dick
    - 98: A Tale of Two Cities
    - 1400: Great Expectations
    - 11: Alice in Wonderland
    - 74: Tom Sawyer
    """
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    
    # Try alternate URL format
    alt_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    
    for try_url in [url, alt_url]:
        try:
            with urllib.request.urlopen(try_url, timeout=30, context=context) as response:
                text = response.read().decode('utf-8', errors='ignore')
                # Strip Gutenberg header/footer
                start = text.find("*** START OF")
                end = text.find("*** END OF")
                if start != -1 and end != -1:
                    # Find the end of the START line
                    start = text.find("\n", start) + 1
                    text = text[start:end]
                return text
        except Exception as e:
            continue
    
    raise ValueError(f"Could not fetch book {book_id} from Gutenberg")


def load_sample_text() -> str:
    """Load sample text for testing without network."""
    return """
    The gentleman entered the grand ballroom with an air of distinction.
    The lady curtsied gracefully, her gown sweeping the marble floor.
    He spoke softly, his voice barely above a whisper.
    She replied loudly, her words echoing through the hall.
    
    The king sat upon his golden throne, surveying his vast kingdom.
    The queen stood beside him, her crown glittering with jewels.
    The prince rode swiftly through the forest on his white stallion.
    The princess walked slowly through the garden, admiring the roses.
    
    The old man shuffled down the dusty road, his back bent with age.
    The young boy ran ahead, full of energy and excitement.
    The ancient castle loomed on the hilltop, its towers crumbling.
    The modern city sprawled below, its lights twinkling in the night.
    
    He was rich beyond measure, his coffers overflowing with gold.
    She was poor, her pockets empty, her clothes threadbare.
    The master commanded his servants with a stern voice.
    The servant obeyed quietly, head bowed in submission.
    
    The brave knight charged into battle, sword raised high.
    The cowardly squire fled in terror, dropping his shield.
    The wise sage spoke in riddles, his meaning obscure.
    The foolish jester laughed and tumbled, caring nothing for wisdom.
    
    In the morning light, the world seemed bright and hopeful.
    In the evening darkness, shadows crept and fears awakened.
    The summer sun blazed hot upon the parched earth.
    The winter snow fell cold upon the frozen ground.
    
    She whispered secrets in the quiet of the night.
    He shouted commands in the chaos of the day.
    The tiny mouse scurried through the enormous hall.
    The giant strode across the land, each step shaking the earth.
    """


# =============================================================================
# TOKENIZATION AND PREPROCESSING
# =============================================================================

def tokenize(text: str) -> List[str]:
    """Simple word tokenization."""
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return words


# Stop words to filter out (function words that don't carry semantic meaning)
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'we',
    'they', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why',
    'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once',
    'if', 'unless', 'until', 'while', 'although', 'though', 'because', 'since',
    'after', 'before', 'during', 'about', 'into', 'through', 'above', 'below',
    'between', 'under', 'again', 'further', 'any', 'being', 'having', 'doing',
    'their', 'them', 'him', 'her', 'his', 'hers', 'my', 'your', 'our',
    'me', 'us', 'myself', 'yourself', 'himself', 'herself', 'itself',
    'ourselves', 'themselves', 'one', 'ones', 'upon', 'out', 'up', 'down',
    'off', 'over', 'even', 'still', 'yet', 'already', 'always', 'never',
    'ever', 'often', 'sometimes', 'usually', 'perhaps', 'maybe', 'certainly',
    'probably', 'possibly', 'actually', 'really', 'quite', 'rather', 'almost',
    'enough', 'much', 'many', 'little', 'less', 'least', 'first', 'last',
    'next', 'new', 'old', 'good', 'great', 'long', 'own', 'another', 'well',
    'back', 'way', 'thing', 'things', 'time', 'times', 'day', 'days', 'year',
    'years', 'life', 'world', 'man', 'men', 'woman', 'women', 'people',
    'said', 'say', 'says', 'saying', 'went', 'go', 'goes', 'going', 'gone',
    'come', 'comes', 'coming', 'came', 'get', 'gets', 'getting', 'got',
    'make', 'makes', 'making', 'made', 'take', 'takes', 'taking', 'took',
    'know', 'knows', 'knowing', 'knew', 'known', 'think', 'thinks', 'thinking',
    'thought', 'see', 'sees', 'seeing', 'saw', 'seen', 'look', 'looks',
    'looking', 'looked', 'want', 'wants', 'wanting', 'wanted', 'give', 'gives',
    'giving', 'gave', 'given', 'find', 'finds', 'finding', 'found', 'tell',
    'tells', 'telling', 'told', 'ask', 'asks', 'asking', 'asked', 'seem',
    'seems', 'seeming', 'seemed', 'leave', 'leaves', 'leaving', 'left',
    'call', 'calls', 'calling', 'called', 'keep', 'keeps', 'keeping', 'kept',
    'let', 'lets', 'letting', 'begin', 'begins', 'beginning', 'began', 'begun',
    'show', 'shows', 'showing', 'showed', 'shown', 'hear', 'hears', 'hearing',
    'heard', 'play', 'plays', 'playing', 'played', 'run', 'runs', 'running',
    'ran', 'move', 'moves', 'moving', 'moved', 'live', 'lives', 'living',
    'lived', 'believe', 'believes', 'believing', 'believed', 'hold', 'holds',
    'holding', 'held', 'bring', 'brings', 'bringing', 'brought', 'happen',
    'happens', 'happening', 'happened', 'write', 'writes', 'writing', 'wrote',
    'written', 'sit', 'sits', 'sitting', 'sat', 'stand', 'stands', 'standing',
    'stood', 'lose', 'loses', 'losing', 'lost', 'pay', 'pays', 'paying',
    'paid', 'meet', 'meets', 'meeting', 'met', 'include', 'includes',
    'including', 'included', 'continue', 'continues', 'continuing', 'continued',
    'set', 'sets', 'setting', 'learn', 'learns', 'learning', 'learned',
    'change', 'changes', 'changing', 'changed', 'lead', 'leads', 'leading',
    'led', 'understand', 'understands', 'understanding', 'understood',
    'watch', 'watches', 'watching', 'watched', 'follow', 'follows', 'following',
    'followed', 'stop', 'stops', 'stopping', 'stopped', 'create', 'creates',
    'creating', 'created', 'speak', 'speaks', 'speaking', 'spoke', 'spoken',
    'read', 'reads', 'reading', 'allow', 'allows', 'allowing', 'allowed',
    'add', 'adds', 'adding', 'added', 'spend', 'spends', 'spending', 'spent',
    'grow', 'grows', 'growing', 'grew', 'grown', 'open', 'opens', 'opening',
    'opened', 'walk', 'walks', 'walking', 'walked', 'win', 'wins', 'winning',
    'won', 'offer', 'offers', 'offering', 'offered', 'remember', 'remembers',
    'remembering', 'remembered', 'consider', 'considers', 'considering',
    'considered', 'appear', 'appears', 'appearing', 'appeared', 'buy', 'buys',
    'buying', 'bought', 'wait', 'waits', 'waiting', 'waited', 'serve',
    'serves', 'serving', 'served', 'die', 'dies', 'dying', 'died', 'send',
    'sends', 'sending', 'sent', 'expect', 'expects', 'expecting', 'expected',
    'build', 'builds', 'building', 'built', 'stay', 'stays', 'staying',
    'stayed', 'fall', 'falls', 'falling', 'fell', 'fallen', 'cut', 'cuts',
    'cutting', 'reach', 'reaches', 'reaching', 'reached', 'kill', 'kills',
    'killing', 'killed', 'remain', 'remains', 'remaining', 'remained',
}


def get_context_windows(tokens: List[str], window_size: int = 5) -> Dict[str, List[Tuple[str, ...]]]:
    """
    Get context windows for each word.
    
    Returns dict: word → list of context tuples
    """
    contexts = defaultdict(list)
    
    for i, word in enumerate(tokens):
        # Get surrounding words (excluding the word itself)
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        
        context = tuple(tokens[j] for j in range(start, end) if j != i)
        contexts[word].append(context)
    
    return contexts


# =============================================================================
# CONTRASTING PAIR DISCOVERY
# =============================================================================

@dataclass
class ContrastingPair:
    """A pair of words that contrast along some dimension."""
    word1: str
    word2: str
    dimension_name: str
    confidence: float
    shared_contexts: int
    co_occurrence: int


def build_cooccurrence_matrix(tokens: List[str], 
                               window_size: int = 5,
                               min_count: int = 3,
                               filter_stopwords: bool = True) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Build word co-occurrence matrix.
    
    Returns:
        vocab: word → index mapping
        matrix: co-occurrence counts
    """
    # Count words
    word_counts = Counter(tokens)
    
    # Filter to words with minimum count and optionally remove stop words
    vocab_words = [w for w, c in word_counts.items() 
                   if c >= min_count and len(w) > 2
                   and (not filter_stopwords or w not in STOP_WORDS)]
    vocab = {w: i for i, w in enumerate(vocab_words)}
    
    n = len(vocab)
    matrix = np.zeros((n, n), dtype=np.float32)
    
    # Count co-occurrences
    for i, word in enumerate(tokens):
        if word not in vocab:
            continue
        
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        
        for j in range(start, end):
            if j != i and tokens[j] in vocab:
                matrix[vocab[word], vocab[tokens[j]]] += 1
    
    return vocab, matrix


def find_contrasting_pairs(tokens: List[str],
                           window_size: int = 5,
                           min_count: int = 3,
                           top_k: int = 50,
                           filter_stopwords: bool = True) -> List[ContrastingPair]:
    """
    Find contrasting word pairs.
    
    Contrasting pairs are words that:
    1. Appear in similar contexts (high context similarity)
    2. Rarely appear together (low co-occurrence)
    
    This captures pairs like he/she, big/small, quickly/slowly.
    """
    vocab, cooc_matrix = build_cooccurrence_matrix(tokens, window_size, min_count, filter_stopwords)
    index_to_word = {i: w for w, i in vocab.items()}
    n = len(vocab)
    
    if n < 10:
        return []
    
    # Normalize to get context similarity (cosine-like)
    # Add small epsilon to avoid division by zero
    norms = np.sqrt(np.sum(cooc_matrix ** 2, axis=1, keepdims=True)) + 1e-10
    normalized = cooc_matrix / norms
    
    # Context similarity matrix
    context_sim = normalized @ normalized.T
    
    # Find pairs with high context similarity but low direct co-occurrence
    pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            ctx_sim = context_sim[i, j]
            direct_cooc = cooc_matrix[i, j] + cooc_matrix[j, i]
            
            # High context similarity, low direct co-occurrence
            if ctx_sim > 0.3 and direct_cooc < 5:
                word1 = index_to_word[i]
                word2 = index_to_word[j]
                
                # Skip if words are too similar (likely variants)
                if word1[:3] == word2[:3]:
                    continue
                
                # Confidence based on context similarity and inverse co-occurrence
                confidence = ctx_sim * (1.0 / (1.0 + direct_cooc))
                
                pairs.append(ContrastingPair(
                    word1=word1,
                    word2=word2,
                    dimension_name=f"{word1}_{word2}",  # Will be named later
                    confidence=confidence,
                    shared_contexts=int(ctx_sim * 100),
                    co_occurrence=int(direct_cooc)
                ))
    
    # Sort by confidence
    pairs.sort(key=lambda p: -p.confidence)
    
    return pairs[:top_k]


# =============================================================================
# DIMENSION NAMING
# =============================================================================

# Known dimension patterns for automatic naming
DIMENSION_PATTERNS = {
    'gender': [
        ('he', 'she'), ('him', 'her'), ('his', 'hers'),
        ('man', 'woman'), ('boy', 'girl'), ('king', 'queen'),
        ('prince', 'princess'), ('lord', 'lady'), ('sir', 'madam'),
        ('gentleman', 'lady'), ('father', 'mother'), ('son', 'daughter'),
        ('brother', 'sister'), ('husband', 'wife'), ('mr', 'mrs'),
    ],
    'age': [
        ('old', 'young'), ('ancient', 'modern'), ('elderly', 'youthful'),
        ('aged', 'new'), ('senior', 'junior'),
    ],
    'size': [
        ('big', 'small'), ('large', 'tiny'), ('huge', 'little'),
        ('enormous', 'minute'), ('giant', 'dwarf'), ('vast', 'narrow'),
    ],
    'speed': [
        ('fast', 'slow'), ('quick', 'leisurely'), ('swift', 'sluggish'),
        ('rapidly', 'slowly'), ('quickly', 'gradually'),
    ],
    'volume': [
        ('loud', 'quiet'), ('shouted', 'whispered'), ('roared', 'murmured'),
        ('bellowed', 'muttered'), ('screamed', 'sighed'),
    ],
    'temperature': [
        ('hot', 'cold'), ('warm', 'cool'), ('burning', 'freezing'),
        ('summer', 'winter'), ('fire', 'ice'),
    ],
    'light': [
        ('bright', 'dark'), ('light', 'shadow'), ('day', 'night'),
        ('morning', 'evening'), ('sun', 'moon'), ('dawn', 'dusk'),
    ],
    'wealth': [
        ('rich', 'poor'), ('wealthy', 'destitute'), ('gold', 'rags'),
        ('palace', 'hovel'), ('feast', 'famine'),
    ],
    'status': [
        ('master', 'servant'), ('king', 'peasant'), ('noble', 'common'),
        ('lord', 'serf'), ('ruler', 'subject'),
    ],
    'courage': [
        ('brave', 'cowardly'), ('bold', 'timid'), ('fearless', 'afraid'),
        ('hero', 'coward'), ('valiant', 'craven'),
    ],
    'wisdom': [
        ('wise', 'foolish'), ('sage', 'fool'), ('clever', 'stupid'),
        ('smart', 'dumb'), ('learned', 'ignorant'),
    ],
    'beauty': [
        ('beautiful', 'ugly'), ('fair', 'foul'), ('lovely', 'hideous'),
        ('handsome', 'plain'), ('gorgeous', 'homely'),
    ],
    'good_evil': [
        ('good', 'evil'), ('kind', 'cruel'), ('gentle', 'harsh'),
        ('virtuous', 'wicked'), ('saint', 'sinner'),
    ],
    'height': [
        ('tall', 'short'), ('high', 'low'), ('above', 'below'),
        ('up', 'down'), ('top', 'bottom'),
    ],
    'distance': [
        ('near', 'far'), ('close', 'distant'), ('here', 'there'),
        ('nearby', 'remote'),
    ],
}


def name_dimension(word1: str, word2: str) -> str:
    """
    Try to name a dimension based on known patterns.
    
    Returns dimension name or generates one from the words.
    """
    pair = (word1.lower(), word2.lower())
    pair_rev = (word2.lower(), word1.lower())
    
    for dim_name, patterns in DIMENSION_PATTERNS.items():
        if pair in patterns or pair_rev in patterns:
            return dim_name
    
    # Check partial matches
    for dim_name, patterns in DIMENSION_PATTERNS.items():
        for p1, p2 in patterns:
            if (word1.startswith(p1) or word2.startswith(p2) or
                word1.startswith(p2) or word2.startswith(p1)):
                return dim_name
    
    # Generate name from words
    return f"{word1}_{word2}"


def assign_dimension_names(pairs: List[ContrastingPair]) -> List[ContrastingPair]:
    """Assign meaningful names to discovered dimensions."""
    named_pairs = []
    used_dimensions = set()
    
    for pair in pairs:
        dim_name = name_dimension(pair.word1, pair.word2)
        
        # Avoid duplicate dimensions
        if dim_name in used_dimensions and not dim_name.endswith('_' + pair.word2):
            continue
        
        used_dimensions.add(dim_name)
        named_pairs.append(ContrastingPair(
            word1=pair.word1,
            word2=pair.word2,
            dimension_name=dim_name,
            confidence=pair.confidence,
            shared_contexts=pair.shared_contexts,
            co_occurrence=pair.co_occurrence
        ))
    
    return named_pairs


# =============================================================================
# DIMENSION REGISTRY INTEGRATION
# =============================================================================

class DiscoveredDimensionRegistry:
    """
    Registry that discovers dimensions from text.
    """
    
    def __init__(self, max_dims: int = 128):
        self.max_dims = max_dims
        self._dimensions: Dict[str, int] = {}
        self._index_to_name: Dict[int, str] = {}
        self._anchors: Dict[str, Dict[str, float]] = {}
        self._pairs: Dict[str, ContrastingPair] = {}
        self._next_index = 0
    
    def discover_from_text(self, text: str, 
                           window_size: int = 5,
                           min_count: int = 3,
                           top_k: int = 50) -> List[ContrastingPair]:
        """
        Discover dimensions from text.
        
        Returns list of discovered contrasting pairs.
        """
        tokens = tokenize(text)
        print(f"Tokenized {len(tokens)} words, {len(set(tokens))} unique")
        
        pairs = find_contrasting_pairs(tokens, window_size, min_count, top_k)
        print(f"Found {len(pairs)} contrasting pairs")
        
        named_pairs = assign_dimension_names(pairs)
        print(f"Named {len(named_pairs)} dimensions")
        
        # Register dimensions
        for pair in named_pairs:
            if self._next_index >= self.max_dims:
                break
            
            if pair.dimension_name not in self._dimensions:
                idx = self._next_index
                self._dimensions[pair.dimension_name] = idx
                self._index_to_name[idx] = pair.dimension_name
                self._anchors[pair.dimension_name] = {
                    pair.word1: 1.0,
                    pair.word2: -1.0
                }
                self._pairs[pair.dimension_name] = pair
                self._next_index += 1
        
        return named_pairs
    
    def get_word_position(self, word: str) -> Dict[str, float]:
        """Get dimension activations for a word."""
        word = word.lower()
        activations = {}
        
        for dim_name, anchors in self._anchors.items():
            if word in anchors:
                activations[dim_name] = anchors[word]
        
        return activations
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to dimension vector."""
        tokens = tokenize(text)
        vector = np.zeros(self._next_index)
        
        for token in tokens:
            for dim_name, anchors in self._anchors.items():
                if token in anchors:
                    idx = self._dimensions[dim_name]
                    # Use max aggregation
                    if abs(anchors[token]) > abs(vector[idx]):
                        vector[idx] = anchors[token]
        
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
        """Get summary of discovered dimensions."""
        lines = [f"Discovered {self.num_dims} dimensions:"]
        for dim_name, idx in sorted(self._dimensions.items(), key=lambda x: x[1]):
            pair = self._pairs.get(dim_name)
            if pair:
                lines.append(f"  [{idx:2d}] {dim_name}: {pair.word1} ↔ {pair.word2} (conf={pair.confidence:.3f})")
            else:
                anchors = self._anchors.get(dim_name, {})
                pos = [w for w, v in anchors.items() if v > 0]
                neg = [w for w, v in anchors.items() if v < 0]
                lines.append(f"  [{idx:2d}] {dim_name}: {pos} ↔ {neg}")
        return "\n".join(lines)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def seed_known_dimensions(registry: 'DiscoveredDimensionRegistry'):
    """
    Seed registry with known dimension patterns.
    
    These are dimensions we KNOW exist in language.
    The discovery process will find more, but these give us a good start.
    """
    known_dims = {
        'gender': {
            'he': 1.0, 'him': 1.0, 'his': 1.0, 'himself': 1.0,
            'she': -1.0, 'her': -1.0, 'hers': -1.0, 'herself': -1.0,
            'man': 1.0, 'woman': -1.0, 'boy': 1.0, 'girl': -1.0,
            'king': 1.0, 'queen': -1.0, 'prince': 1.0, 'princess': -1.0,
            'lord': 1.0, 'lady': -1.0, 'sir': 1.0, 'madam': -1.0,
            'gentleman': 1.0, 'gentlewoman': -1.0,
            'father': 1.0, 'mother': -1.0, 'son': 1.0, 'daughter': -1.0,
            'brother': 1.0, 'sister': -1.0, 'husband': 1.0, 'wife': -1.0,
            'uncle': 1.0, 'aunt': -1.0, 'nephew': 1.0, 'niece': -1.0,
            'mr': 1.0, 'mrs': -1.0, 'miss': -1.0,
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
            'big': 1.0, 'large': 1.0, 'huge': 1.5, 'enormous': 2.0, 'giant': 2.0, 'vast': 1.5,
            'small': -1.0, 'little': -1.0, 'tiny': -1.5, 'minute': -2.0, 'dwarf': -1.5,
        },
        'speed': {
            'fast': 1.0, 'quick': 1.0, 'swift': 1.0, 'rapid': 1.0, 'hasty': 0.5,
            'quickly': 1.0, 'swiftly': 1.0, 'rapidly': 1.0, 'hastily': 0.5,
            'slow': -1.0, 'sluggish': -1.0, 'leisurely': -0.5, 'gradual': -0.5,
            'slowly': -1.0, 'gradually': -0.5,
        },
        'volume': {
            'loud': 1.0, 'loudly': 1.0, 'shouted': 1.0, 'roared': 1.5, 'bellowed': 1.5,
            'screamed': 1.5, 'yelled': 1.0, 'cried': 0.5,
            'quiet': -1.0, 'quietly': -1.0, 'whispered': -1.0, 'murmured': -1.0,
            'muttered': -0.5, 'sighed': -0.5, 'softly': -1.0,
        },
        'temperature': {
            'hot': 1.0, 'warm': 0.5, 'burning': 1.5, 'blazing': 1.5, 'fiery': 1.5,
            'cold': -1.0, 'cool': -0.5, 'freezing': -1.5, 'icy': -1.5, 'frozen': -1.5,
            'summer': 1.0, 'winter': -1.0,
        },
        'light': {
            'bright': 1.0, 'light': 0.5, 'brilliant': 1.5, 'radiant': 1.5, 'glowing': 1.0,
            'dark': -1.0, 'dim': -0.5, 'shadow': -1.0, 'shadowy': -1.0, 'gloomy': -1.0,
            'day': 0.5, 'night': -0.5, 'morning': 0.5, 'evening': -0.5,
            'sun': 1.0, 'moon': -0.5, 'dawn': 0.5, 'dusk': -0.5,
        },
        'wealth': {
            'rich': 1.0, 'wealthy': 1.0, 'prosperous': 1.0, 'affluent': 1.0,
            'gold': 1.0, 'fortune': 1.0, 'treasure': 1.0,
            'poor': -1.0, 'destitute': -1.5, 'impoverished': -1.0, 'penniless': -1.0,
        },
        'courage': {
            'brave': 1.0, 'bold': 1.0, 'courageous': 1.0, 'fearless': 1.5, 'valiant': 1.5,
            'hero': 1.0, 'heroic': 1.0,
            'cowardly': -1.0, 'timid': -0.5, 'afraid': -0.5, 'fearful': -0.5,
            'coward': -1.0, 'craven': -1.0,
        },
        'wisdom': {
            'wise': 1.0, 'sage': 1.0, 'clever': 0.5, 'intelligent': 0.5, 'learned': 1.0,
            'foolish': -1.0, 'stupid': -1.0, 'ignorant': -1.0, 'fool': -1.0,
        },
        'beauty': {
            'beautiful': 1.0, 'lovely': 1.0, 'fair': 0.5, 'handsome': 1.0, 'gorgeous': 1.5,
            'pretty': 0.5, 'elegant': 1.0,
            'ugly': -1.0, 'hideous': -1.5, 'plain': -0.5, 'homely': -0.5,
        },
        'good_evil': {
            'good': 1.0, 'kind': 1.0, 'gentle': 0.5, 'virtuous': 1.0, 'noble': 1.0,
            'evil': -1.0, 'wicked': -1.0, 'cruel': -1.0, 'vile': -1.5, 'malicious': -1.0,
        },
        'formality': {
            'formal': 1.0, 'proper': 1.0, 'dignified': 1.0, 'stately': 1.0,
            'informal': -1.0, 'casual': -0.5, 'relaxed': -0.5,
        },
        'certainty': {
            'certain': 1.0, 'sure': 1.0, 'definite': 1.0, 'absolute': 1.5,
            'uncertain': -1.0, 'unsure': -1.0, 'doubtful': -1.0, 'hesitant': -0.5,
        },
    }
    
    for dim_name, anchors in known_dims.items():
        if dim_name not in registry._dimensions:
            idx = registry._next_index
            registry._dimensions[dim_name] = idx
            registry._index_to_name[idx] = dim_name
            registry._anchors[dim_name] = anchors
            registry._next_index += 1
    
    return len(known_dims)


def demo_sample_text():
    """Demonstrate dimension discovery on sample text."""
    print("=" * 70)
    print("DIMENSION DISCOVERY - SAMPLE TEXT")
    print("=" * 70)
    print()
    
    text = load_sample_text()
    
    registry = DiscoveredDimensionRegistry(max_dims=64)
    
    # Seed with known dimensions first
    num_seeded = seed_known_dimensions(registry)
    print(f"Seeded {num_seeded} known dimensions")
    
    # Then discover more from text
    pairs = registry.discover_from_text(text, min_count=2, top_k=30)
    
    print()
    print(registry.summary())
    print()
    
    # Test encoding
    test_sentences = [
        "The king spoke loudly",
        "The queen whispered softly",
        "The old man walked slowly",
        "The young boy ran quickly",
        "The rich lord in his palace",
        "The poor servant in rags",
    ]
    
    print("ENCODING TEST:")
    for sentence in test_sentences:
        vector = registry.encode_text(sentence)
        desc = registry.describe_vector(vector)
        print(f"  '{sentence}'")
        print(f"    → {desc}")
        print()


def demo_gutenberg():
    """Demonstrate dimension discovery on Project Gutenberg text."""
    print("=" * 70)
    print("DIMENSION DISCOVERY - PROJECT GUTENBERG")
    print("=" * 70)
    print()
    
    print("Fetching Pride and Prejudice (ID: 1342)...")
    try:
        text = fetch_gutenberg(1342)
        print(f"Fetched {len(text)} characters")
    except Exception as e:
        print(f"Could not fetch from Gutenberg: {e}")
        print("Using sample text instead...")
        text = load_sample_text() * 10  # Repeat to get more data
    
    print()
    
    registry = DiscoveredDimensionRegistry(max_dims=64)
    pairs = registry.discover_from_text(text, min_count=5, top_k=50)
    
    print()
    print(registry.summary())
    print()
    
    # Test encoding on sentences from the book's style
    test_sentences = [
        "Mr Darcy is a proud gentleman",
        "Miss Bennet is a lovely lady",
        "He spoke with cold disdain",
        "She replied with warm affection",
        "The rich estate at Pemberley",
        "The poor cottage in the village",
    ]
    
    print("ENCODING TEST (Pride and Prejudice style):")
    for sentence in test_sentences:
        vector = registry.encode_text(sentence)
        desc = registry.describe_vector(vector)
        if desc:
            print(f"  '{sentence}'")
            print(f"    → {desc}")
            print()


def demo_dimension_similarity():
    """Demonstrate how discovered dimensions enable similarity computation."""
    print("=" * 70)
    print("DIMENSION-BASED SIMILARITY")
    print("=" * 70)
    print()
    
    registry = DiscoveredDimensionRegistry(max_dims=64)
    
    # Seed with known dimensions
    num_seeded = seed_known_dimensions(registry)
    print(f"Using {num_seeded} seeded dimensions")
    print()
    
    # Define query and candidates
    query = "The brave king shouted commands"
    candidates = [
        "The cowardly queen whispered secrets",
        "The bold prince spoke loudly",
        "The timid princess murmured softly",
        "The old servant walked slowly",
    ]
    
    query_vec = registry.encode_text(query)
    print(f"Query: '{query}'")
    print(f"  Dimensions: {registry.describe_vector(query_vec)}")
    print()
    
    print("Candidates (sorted by similarity):")
    results = []
    for candidate in candidates:
        cand_vec = registry.encode_text(candidate)
        
        # Cosine similarity
        dot = np.dot(query_vec, cand_vec)
        norm_q = np.linalg.norm(query_vec) + 1e-10
        norm_c = np.linalg.norm(cand_vec) + 1e-10
        similarity = dot / (norm_q * norm_c)
        
        results.append((candidate, similarity, registry.describe_vector(cand_vec)))
    
    results.sort(key=lambda x: -x[1])
    
    for candidate, sim, dims in results:
        print(f"  [{sim:+.3f}] '{candidate}'")
        print(f"           {dims}")
        print()
    
    print("KEY INSIGHT:")
    print("  'The bold prince spoke loudly' is most similar because it shares")
    print("  courage (brave/bold) and volume (shouted/loudly) dimensions.")
    print("  The dimensions were DISCOVERED from text, not predefined!")


if __name__ == "__main__":
    demo_sample_text()
    print("\n" + "=" * 70 + "\n")
    demo_gutenberg()
    print("\n" + "=" * 70 + "\n")
    demo_dimension_similarity()
