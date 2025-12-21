#!/usr/bin/env python3
"""
Generalized Holographic Q&A System

This version automatically extracts semantic seeds from ANY source text,
making it generalizable to any book or corpus.

Key innovations:
1. Auto-extract named entities (characters, places, things) from text
2. Auto-build semantic clusters based on co-occurrence
3. Use universal question gap patterns (these ARE generalizable)
4. No hardcoded domain knowledge required

Usage:
    python holographic_qa_general.py                    # Default: Moby Dick
    python holographic_qa_general.py --book white_fang  # Jack London's White Fang
    python holographic_qa_general.py --url <gutenberg_url>  # Any Gutenberg book
"""

import sys
import os
import re
import argparse
import urllib.request
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional
import numpy as np

PHI = (1 + np.sqrt(5)) / 2

# ============================================================================
# UNIVERSAL QUESTION GAPS - These are language-universal, not domain-specific
# ============================================================================

UNIVERSAL_GAPS = {
    'IDENTITY': {
        'patterns': ['who is', 'who was', 'who are', 'who were', 'tell me about', 'describe'],
        'fill_words': {'is', 'was', 'named', 'called', 'known', 'man', 'woman', 'person', 'he', 'she'},
    },
    'EVENT': {
        'patterns': ['what happened', 'what occurs', 'what took place', 'did what'],
        'fill_words': {'happened', 'occurred', 'died', 'killed', 'destroyed', 'end', 'began', 'started'},
    },
    'DEFINITION': {
        'patterns': ['what is', 'what was', 'what are', 'what does', 'define', 'meaning of'],
        'fill_words': {'is', 'was', 'means', 'refers', 'defined', 'called', 'type', 'kind'},
    },
    'TIME': {
        'patterns': ['when did', 'when was', 'when is', 'what year', 'what date', 'how long'],
        'fill_words': {'year', 'date', 'time', 'day', 'month', 'ago', 'in', 'on', 'at', 'before', 'after'},
    },
    'LOCATION': {
        'patterns': ['where is', 'where was', 'where did', 'what place', 'location of'],
        'fill_words': {'in', 'at', 'near', 'place', 'located', 'found', 'from', 'to'},
    },
    'REASON': {
        'patterns': ['why did', 'why does', 'why is', 'why was', 'reason', 'cause of'],
        'fill_words': {'because', 'since', 'due', 'reason', 'cause', 'therefore', 'so'},
    },
    'METHOD': {
        'patterns': ['how did', 'how does', 'how do', 'how to', 'how is', 'method', 'way to'],
        'fill_words': {'by', 'through', 'using', 'method', 'way', 'process'},
    },
}

# Universal stop words for any English text
STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between',
    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
    'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'would', 'should', 'could', 'ought',
    'now', 'then', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn',
    'weren', 'won', 'wouldn', 'said', 'upon', 'one', 'two', 'first',
}


class TextAnalyzer:
    """
    Analyzes source text to automatically extract:
    1. Named entities (characters, places, things)
    2. Semantic clusters based on co-occurrence
    3. Key descriptive sentences
    """
    
    def __init__(self):
        self.word_freq = Counter()
        self.capitalized_freq = Counter()  # Likely proper nouns
        self.cooccurrence = defaultdict(Counter)  # Word co-occurrence
        self.sentences = []
    
    def analyze(self, text: str) -> Dict:
        """Analyze text and extract semantic structure."""
        # Clean and split into sentences
        self.sentences = self._extract_sentences(text)
        
        # Count word frequencies
        for sentence in self.sentences:
            words = self._tokenize(sentence)
            
            # Track capitalized words (likely names/places)
            for word in re.findall(r'\b[A-Z][a-z]+\b', sentence):
                word_lower = word.lower()
                if word_lower not in STOP_WORDS and len(word_lower) > 2:
                    self.capitalized_freq[word_lower] += 1
            
            # Track all content words
            content_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
            for word in content_words:
                self.word_freq[word] += 1
            
            # Track co-occurrence (words in same sentence)
            for i, w1 in enumerate(content_words):
                for w2 in content_words[i+1:]:
                    self.cooccurrence[w1][w2] += 1
                    self.cooccurrence[w2][w1] += 1
        
        return self._build_semantic_clusters()
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract clean sentences from text."""
        # Remove Gutenberg header/footer
        start_markers = ['CHAPTER 1', 'Chapter 1', 'CHAPTER I', 'Chapter I', '*** START']
        end_markers = ['*** END', 'End of the Project', 'End of Project']
        
        for marker in start_markers:
            idx = text.find(marker)
            if idx != -1:
                text = text[idx:]
                break
        
        for marker in end_markers:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
                break
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean sentences
        clean = []
        for s in sentences:
            s = re.sub(r'\s+', ' ', s).strip()
            if 30 < len(s) < 500 and not s.startswith('"'):
                clean.append(s)
        
        return clean
    
    def _build_semantic_clusters(self) -> Dict[str, List[str]]:
        """Build semantic clusters from co-occurrence patterns."""
        clusters = {}
        
        # Get top entities (likely characters, places, key concepts)
        # Prioritize capitalized words (proper nouns)
        top_entities = [w for w, c in self.capitalized_freq.most_common(50) if c >= 5]
        
        # Find the MOST frequent entity (likely the protagonist) to exclude from others' clusters
        protagonist = top_entities[0] if top_entities else None
        
        # Build a cluster for each top entity
        used_words = set()
        for entity in top_entities[:25]:
            # Find words that co-occur frequently with this entity
            # BUT exclude the protagonist from other clusters (it's everywhere)
            cooccur = self.cooccurrence[entity]
            
            # Get related words, excluding the protagonist and already-used words
            related = []
            for w, c in cooccur.most_common(20):
                if c >= 3 and w != protagonist and w not in used_words:
                    # Compute specificity: how unique is this co-occurrence?
                    # Words that appear with EVERYONE are less useful
                    total_cooccur = sum(self.cooccurrence[w].values())
                    specificity = c / (total_cooccur + 1)
                    if specificity > 0.05 or c >= 10:  # Either specific or very frequent
                        related.append(w)
                        if len(related) >= 6:
                            break
            
            if related:
                cluster_name = entity.upper()
                clusters[cluster_name] = [entity] + related
                used_words.add(entity)
                used_words.update(related)
        
        # Add concept clusters for important non-entity words
        concept_words = ['wolf', 'dog', 'wild', 'kill', 'fight', 'love', 'fear', 
                         'master', 'god', 'life', 'death', 'mother', 'father',
                         'camp', 'forest', 'snow', 'cold', 'hunger', 'pack']
        
        for word in concept_words:
            if word in self.word_freq and self.word_freq[word] >= 10 and word not in used_words:
                cooccur = self.cooccurrence[word]
                related = [w for w, c in cooccur.most_common(8) 
                           if c >= 3 and w not in used_words and w != protagonist][:5]
                if related:
                    clusters[word.upper()] = [word] + related
                    used_words.add(word)
        
        return clusters
    
    def extract_facts(self) -> List[Tuple[str, str]]:
        """Extract fact-like sentences, prioritizing descriptive ones."""
        facts = []
        scored_facts = []
        
        for sentence in self.sentences:
            words = sentence.lower().split()
            
            # Skip questions
            if sentence.strip().endswith('?'):
                continue
            
            # Score sentences by how "fact-like" they are
            score = 0
            
            # Descriptive verbs
            if any(v in words for v in ['is', 'was', 'are', 'were']):
                score += 2
            if any(v in words for v in ['called', 'named', 'known']):
                score += 3
            
            # Contains a capitalized word (likely a name)
            if re.search(r'\b[A-Z][a-z]+\b', sentence):
                score += 1
            
            # Sentence structure: starts with a name (likely a description)
            if re.match(r'^[A-Z][a-z]+\s+(is|was|had|has)\b', sentence):
                score += 3
            
            # Contains relationship words
            if any(w in words for w in ['mother', 'father', 'son', 'daughter', 'master', 'owner']):
                score += 2
            
            # Contains location/time words
            if any(w in words for w in ['where', 'place', 'land', 'country', 'year', 'time']):
                score += 1
            
            # Contains action/event words
            if any(w in words for w in ['killed', 'died', 'born', 'became', 'discovered', 'found']):
                score += 2
            
            if score > 0:
                scored_facts.append((score, sentence))
        
        # Sort by score (highest first) and take top 1000
        scored_facts.sort(key=lambda x: -x[0])
        
        for score, sentence in scored_facts[:1000]:
            facts.append((sentence, "extracted"))
        
        return facts


class GeneralizedHolographicQA:
    """
    Holographic Q&A system that works with ANY text.
    
    Seeds are automatically extracted from the source material.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.facts: List[Tuple[str, str]] = []
        self.word_positions: Dict[str, np.ndarray] = {}
        self.seed_positions: Dict[str, np.ndarray] = {}
        self.semantic_clusters: Dict[str, List[str]] = {}
        self.book_title = "Unknown"
    
    def ingest_text(self, text: str, title: str = "Unknown"):
        """Ingest a text and automatically build semantic structure."""
        self.book_title = title
        
        print(f"  Analyzing text structure...")
        analyzer = TextAnalyzer()
        self.semantic_clusters = analyzer.analyze(text)
        
        print(f"  Found {len(self.semantic_clusters)} semantic clusters:")
        for name, words in list(self.semantic_clusters.items())[:10]:
            print(f"    {name}: {words[:5]}...")
        if len(self.semantic_clusters) > 10:
            print(f"    ... and {len(self.semantic_clusters) - 10} more")
        
        # Initialize seed positions
        self._init_seeds()
        
        # Extract and store facts
        print(f"  Extracting facts...")
        facts = analyzer.extract_facts()
        for fact, source in facts:
            self.store(fact, source)
        
        print(f"  Stored {len(self.facts)} facts")
    
    def _init_seeds(self):
        """Initialize seed positions from extracted clusters."""
        for seed_name in self.semantic_clusters.keys():
            np.random.seed(hash(seed_name) % (2**32))
            pos = np.random.randn(self.dim)
            pos = pos / np.linalg.norm(pos) * PHI
            self.seed_positions[seed_name] = pos
            np.random.seed(None)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def _get_word_position(self, word: str) -> np.ndarray:
        if word in self.word_positions:
            return self.word_positions[word]
        
        # Check if word belongs to any cluster
        for seed_name, seed_words in self.semantic_clusters.items():
            if word in seed_words:
                np.random.seed(hash(word) % (2**32))
                offset = np.random.randn(self.dim) * 0.1
                np.random.seed(None)
                pos = self.seed_positions[seed_name] + offset
                self.word_positions[word] = pos
                return pos
        
        # Unknown word - random position
        np.random.seed(hash(word) % (2**32))
        pos = np.random.randn(self.dim) * 0.3
        np.random.seed(None)
        self.word_positions[word] = pos
        return pos
    
    def _detect_gap_type(self, text: str) -> Tuple[Optional[str], Set[str]]:
        """Detect question type using universal patterns."""
        text_lower = text.lower()
        
        for gap_type, info in UNIVERSAL_GAPS.items():
            for pattern in info['patterns']:
                if pattern in text_lower:
                    return gap_type, info['fill_words']
        
        return None, set()
    
    def _extract_content_words(self, text: str) -> Set[str]:
        words = set(self._tokenize(text))
        return words - STOP_WORDS
    
    def _extract_subject(self, text: str) -> Set[str]:
        """Extract the subject of a sentence."""
        words = self._tokenize(text)
        question_words = {'who', 'what', 'when', 'where', 'why', 'how', 'which'}
        
        if words and words[0] in question_words:
            for i, word in enumerate(words):
                if word in ['is', 'was', 'are', 'were', 'does', 'did']:
                    subject_words = set(words[i+1:])
                    return subject_words - {'the', 'a', 'an'}
        
        for i, word in enumerate(words):
            if word in ['is', 'was', 'are', 'were', 'has', 'had', 'does', 'did']:
                subject_words = set(words[:i])
                return subject_words - STOP_WORDS
        
        return set()
    
    def _encode_content(self, text: str) -> np.ndarray:
        content_words = self._extract_content_words(text)
        if not content_words:
            return np.zeros(self.dim)
        
        position = np.zeros(self.dim)
        for word in content_words:
            position += self._get_word_position(word)
        
        return position / len(content_words)
    
    def _compute_holographic_score(self, query: str, candidate: str) -> Tuple[float, Dict]:
        """Compute holographic match score."""
        # Content similarity
        query_pos = self._encode_content(query)
        candidate_pos = self._encode_content(candidate)
        
        norm_q = np.linalg.norm(query_pos)
        norm_c = np.linalg.norm(candidate_pos)
        
        if norm_q < 1e-8 or norm_c < 1e-8:
            content_sim = 0.0
        else:
            content_sim = np.dot(query_pos, candidate_pos) / (norm_q * norm_c)
        
        # Word overlap
        query_content = self._extract_content_words(query)
        candidate_content = self._extract_content_words(candidate)
        word_overlap = len(query_content & candidate_content) / max(len(query_content), 1)
        
        content_score = 0.5 * content_sim + 0.5 * word_overlap
        
        # Gap fill
        gap_type, fill_words = self._detect_gap_type(query)
        candidate_words = set(self._tokenize(candidate))
        
        if fill_words:
            fill_overlap = len(candidate_words & fill_words)
            gap_fill = fill_overlap / len(fill_words)
        else:
            gap_fill = 0.0
        
        # Subject match boost
        query_subject = self._extract_subject(query)
        candidate_subject = self._extract_subject(candidate)
        
        if query_subject and candidate_subject and (query_subject & candidate_subject):
            gap_fill = min(gap_fill * 1.5, 1.0)
        
        # Holographic score
        gap_multiplier = 0.3 + 0.7 * gap_fill
        holographic_score = content_score * gap_multiplier
        
        return holographic_score, {
            'content_sim': content_sim,
            'word_overlap': word_overlap,
            'gap_type': gap_type,
            'gap_fill': gap_fill,
        }
    
    def store(self, text: str, source: str = "unknown"):
        self.facts.append((text, source))
    
    def query(self, question: str, top_k: int = 3) -> List[Tuple[str, str, float, Dict]]:
        results = []
        for text, source in self.facts:
            score, details = self._compute_holographic_score(question, text)
            results.append((text, source, score, details))
        results.sort(key=lambda x: -x[2])
        return results[:top_k]
    
    def answer(self, question: str) -> Tuple[str, float, Dict]:
        results = self.query(question, top_k=1)
        if results:
            text, source, score, details = results[0]
            return text, score, details
        return "I don't know.", 0.0, {}


# ============================================================================
# BOOK SOURCES - Project Gutenberg URLs
# ============================================================================

GUTENBERG_BOOKS = {
    'moby_dick': {
        'url': 'https://www.gutenberg.org/files/2701/2701-0.txt',
        'title': 'Moby Dick',
        'author': 'Herman Melville',
    },
    'white_fang': {
        'url': 'https://www.gutenberg.org/files/910/910-0.txt',
        'title': 'White Fang',
        'author': 'Jack London',
    },
    'pride_prejudice': {
        'url': 'https://www.gutenberg.org/files/1342/1342-0.txt',
        'title': 'Pride and Prejudice',
        'author': 'Jane Austen',
    },
    'frankenstein': {
        'url': 'https://www.gutenberg.org/files/84/84-0.txt',
        'title': 'Frankenstein',
        'author': 'Mary Shelley',
    },
    'dracula': {
        'url': 'https://www.gutenberg.org/files/345/345-0.txt',
        'title': 'Dracula',
        'author': 'Bram Stoker',
    },
    'sherlock': {
        'url': 'https://www.gutenberg.org/files/1661/1661-0.txt',
        'title': 'The Adventures of Sherlock Holmes',
        'author': 'Arthur Conan Doyle',
    },
    'alice': {
        'url': 'https://www.gutenberg.org/files/11/11-0.txt',
        'title': "Alice's Adventures in Wonderland",
        'author': 'Lewis Carroll',
    },
    'war_worlds': {
        'url': 'https://www.gutenberg.org/files/36/36-0.txt',
        'title': 'The War of the Worlds',
        'author': 'H.G. Wells',
    },
}


def download_book(url: str, cache_name: str = None) -> str:
    """Download a book from Project Gutenberg."""
    if cache_name:
        cache_path = f"/tmp/{cache_name}.txt"
        if os.path.exists(cache_path):
            print(f"  Loading from cache...")
            with open(cache_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    
    print(f"  Downloading from {url}...")
    with urllib.request.urlopen(url) as response:
        text = response.read().decode('utf-8', errors='ignore')
    
    if cache_name:
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    return text


def main():
    parser = argparse.ArgumentParser(description='Generalized Holographic Q&A System')
    parser.add_argument('--book', type=str, default='moby_dick',
                        help=f'Book to load: {", ".join(GUTENBERG_BOOKS.keys())}')
    parser.add_argument('--url', type=str, default=None,
                        help='Custom Gutenberg URL')
    parser.add_argument('--title', type=str, default='Custom Book',
                        help='Title for custom URL')
    parser.add_argument('--list', action='store_true',
                        help='List available books')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable books:")
        for key, info in GUTENBERG_BOOKS.items():
            print(f"  {key:20s} - {info['title']} by {info['author']}")
        return
    
    print("=" * 70)
    print("  GENERALIZED HOLOGRAPHIC Q&A SYSTEM")
    print("  Auto-extracts semantic structure from ANY text")
    print("=" * 70)
    
    # Load book
    if args.url:
        url = args.url
        title = args.title
        cache_name = None
    else:
        if args.book not in GUTENBERG_BOOKS:
            print(f"Unknown book: {args.book}")
            print(f"Available: {', '.join(GUTENBERG_BOOKS.keys())}")
            return
        
        book_info = GUTENBERG_BOOKS[args.book]
        url = book_info['url']
        title = f"{book_info['title']} by {book_info['author']}"
        cache_name = args.book
    
    print(f"\n[1] Loading: {title}")
    text = download_book(url, cache_name)
    print(f"    Downloaded {len(text):,} characters")
    
    # Initialize and ingest
    print(f"\n[2] Building holographic Q&A system...")
    qa = GeneralizedHolographicQA(dim=64)
    qa.ingest_text(text, title)
    
    # Interactive mode
    print("\n" + "=" * 70)
    print(f"  INTERACTIVE Q&A: {title}")
    print("  Ask questions about the book!")
    print("  Commands: 'quit', 'top3', 'clusters', 'help'")
    print("=" * 70)
    
    show_top3 = False
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.lower() == 'top3':
            show_top3 = not show_top3
            print(f"  [Top 3 mode: {'ON' if show_top3 else 'OFF'}]")
            continue
        
        if user_input.lower() == 'clusters':
            print("\n  Discovered semantic clusters:")
            for name, words in list(qa.semantic_clusters.items())[:15]:
                print(f"    {name}: {words[:5]}")
            continue
        
        if user_input.lower() == 'help':
            print("""
  Example questions:
    - Who is [character name]?
    - What is [thing]?
    - What happened to [character]?
    - Where did [event] take place?
    - Why did [character] do [action]?
    - Tell me about [topic]
    
  Commands:
    - top3     : Toggle showing top 3 matches
    - clusters : Show discovered semantic clusters
    - quit     : Exit
            """)
            continue
        
        if show_top3:
            results = qa.query(user_input, top_k=3)
            print(f"\n  Top 3 matches:")
            for i, (text, source, score, details) in enumerate(results):
                preview = text[:150] + "..." if len(text) > 150 else text
                gap = details.get('gap_type', 'None')
                print(f"\n  [{i+1}] Score: {score:.3f} (gap={gap})")
                print(f"      {preview}")
        else:
            answer, score, details = qa.answer(user_input)
            gap_type = details.get('gap_type', 'None')
            gap_fill = details.get('gap_fill', 0)
            
            print(f"   [Gap: {gap_type}, Fill: {gap_fill:.2f}, Score: {score:.3f}]")
            
            if len(answer) > 300:
                answer = answer[:300] + "..."
            print(f"Book: {answer}")


if __name__ == "__main__":
    main()
