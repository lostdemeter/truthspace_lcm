#!/usr/bin/env python3
"""
Holographic Q&A System

The key insight: A question is not just words - it is words + HOLES.
The holes define what kind of answer we need.

In holographic stereoscopy:
  I_L = I - αE    (left eye - has gap)
  I_R = I + αE    (right eye - fills gap)
  
  The GAP between views IS the depth information.

In Q&A:
  Question = Content + Gap (what's missing)
  Answer   = Content + Fill (provides what's missing)
  
  Match = Answer that FILLS the question's gap

Instead of matching on word PRESENCE, we match on:
1. Content overlap (the subject matter)
2. Gap filling (answer provides what question lacks)
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import re

PHI = (1 + np.sqrt(5)) / 2


# Question type patterns - what KIND of gap does each question have?
QUESTION_GAPS = {
    'WHO': {
        'patterns': ['who is', 'who was', 'who are', 'who were', 'tell me about'],
        'gap_type': 'IDENTITY',
        'fill_words': ['is', 'was', 'named', 'called', 'known', 'captain', 'man', 'woman', 'person'],
    },
    'WHAT': {
        'patterns': ['what is', 'what was', 'what are', 'what does', 'define', 'describe'],
        'gap_type': 'DEFINITION',
        'fill_words': ['is', 'means', 'refers', 'defined', 'called', 'type', 'kind'],
    },
    'WHEN': {
        'patterns': ['when did', 'when was', 'when is', 'what year', 'what date', 'what time'],
        'gap_type': 'TIME',
        'fill_words': ['year', 'date', 'time', 'day', 'month', 'century', 'ago', 'in', 'on', 'at'],
    },
    'WHERE': {
        'patterns': ['where is', 'where was', 'where did', 'what place', 'location'],
        'gap_type': 'LOCATION',
        'fill_words': ['in', 'at', 'near', 'city', 'country', 'place', 'located', 'found'],
    },
    'WHY': {
        'patterns': ['why did', 'why does', 'why is', 'reason', 'cause'],
        'gap_type': 'REASON',
        'fill_words': ['because', 'since', 'due', 'reason', 'cause', 'therefore', 'so'],
    },
    'HOW': {
        'patterns': ['how did', 'how does', 'how to', 'how is', 'method', 'way to'],
        'gap_type': 'METHOD',
        'fill_words': ['by', 'through', 'using', 'method', 'way', 'process', 'step'],
    },
}

# Content seeds - what the question is ABOUT
CONTENT_SEEDS = {
    'PERSON': ['captain', 'ahab', 'ishmael', 'queequeg', 'starbuck', 'man', 'woman', 'he', 'she'],
    'WHALE': ['whale', 'moby', 'dick', 'leviathan', 'sperm', 'white', 'creature'],
    'SHIP': ['ship', 'pequod', 'vessel', 'boat', 'deck', 'mast', 'sail'],
    'SEA': ['sea', 'ocean', 'water', 'wave', 'deep', 'voyage'],
    'ACTION': ['hunt', 'chase', 'kill', 'harpoon', 'pursue', 'attack'],
    'DEATH': ['death', 'die', 'dead', 'doom', 'fate', 'destruction', 'end'],
}


class HolographicQA:
    """
    Q&A system that matches based on gap-filling, not just word overlap.
    """
    
    def __init__(self, dim: int = 32):
        self.dim = dim
        self.facts: List[Tuple[str, str]] = []  # (text, id)
        self.word_positions: Dict[str, np.ndarray] = {}
        self.seed_positions: Dict[str, np.ndarray] = {}
        
        self._init_seeds()
    
    def _init_seeds(self):
        """Initialize seed positions for content types."""
        all_seeds = {**CONTENT_SEEDS}
        for seed_name in all_seeds.keys():
            np.random.seed(hash(seed_name) % (2**32))
            pos = np.random.randn(self.dim)
            pos = pos / np.linalg.norm(pos) * PHI
            self.seed_positions[seed_name] = pos
            np.random.seed(None)
        
        # Also create positions for gap types
        for q_type, info in QUESTION_GAPS.items():
            gap_name = info['gap_type']
            np.random.seed(hash(gap_name) % (2**32))
            pos = np.random.randn(self.dim)
            pos = pos / np.linalg.norm(pos) * PHI
            self.seed_positions[gap_name] = pos
            np.random.seed(None)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def _get_word_position(self, word: str) -> np.ndarray:
        if word in self.word_positions:
            return self.word_positions[word]
        
        # Check if word belongs to a seed
        for seed_name, seed_words in CONTENT_SEEDS.items():
            if word in seed_words:
                np.random.seed(hash(word) % (2**32))
                offset = np.random.randn(self.dim) * 0.1
                np.random.seed(None)
                pos = self.seed_positions[seed_name] + offset
                self.word_positions[word] = pos
                return pos
        
        # Unknown word
        np.random.seed(hash(word) % (2**32))
        pos = np.random.randn(self.dim) * 0.3
        np.random.seed(None)
        self.word_positions[word] = pos
        return pos
    
    def _detect_question_type(self, text: str) -> Tuple[Optional[str], Set[str]]:
        """
        Detect what type of question this is and what gap it has.
        Returns (gap_type, fill_words_needed)
        """
        text_lower = text.lower()
        
        for q_type, info in QUESTION_GAPS.items():
            for pattern in info['patterns']:
                if pattern in text_lower:
                    return info['gap_type'], set(info['fill_words'])
        
        return None, set()
    
    def _extract_content_words(self, text: str) -> Set[str]:
        """Extract content words (what the text is ABOUT)."""
        words = set(self._tokenize(text))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'to', 'of', 'and', 'in', 'on', 'at', 'for', 'with', 'by',
                      'what', 'when', 'where', 'who', 'how', 'why', 'which',
                      'do', 'does', 'did', 'i', 'you', 'it', 'that', 'this',
                      'me', 'my', 'about', 'tell'}
        
        return words - stop_words
    
    def _encode_content(self, text: str) -> np.ndarray:
        """Encode the CONTENT of text (what it's about)."""
        content_words = self._extract_content_words(text)
        if not content_words:
            return np.zeros(self.dim)
        
        position = np.zeros(self.dim)
        for word in content_words:
            position += self._get_word_position(word)
        
        return position / len(content_words)
    
    def _extract_subject(self, text: str) -> Set[str]:
        """Extract the likely subject of a sentence (first noun phrase)."""
        words = self._tokenize(text)
        
        # For questions, the subject comes AFTER the question word and verb
        # "Who is Captain Ahab?" → subject is "Captain Ahab"
        question_words = {'who', 'what', 'when', 'where', 'why', 'how', 'which'}
        
        if words and words[0] in question_words:
            # Find the verb, subject comes after
            for i, word in enumerate(words):
                if word in ['is', 'was', 'are', 'were', 'does', 'did']:
                    # Subject is everything after the verb
                    subject_words = set(words[i+1:])
                    stop_words = {'the', 'a', 'an'}
                    return subject_words - stop_words
        
        # For statements, subject is usually before "is/was/are/were"
        subject_words = set()
        for i, word in enumerate(words):
            if word in ['is', 'was', 'are', 'were', 'has', 'had', 'does', 'did']:
                subject_words = set(words[:i])
                break
        
        stop_words = {'the', 'a', 'an', 'who', 'what', 'when', 'where', 'why', 'how'}
        return subject_words - stop_words
    
    def _compute_gap_fill_score(self, query: str, candidate: str) -> float:
        """
        Compute how well the candidate FILLS the gap in the query.
        
        This is the holographic insight:
        - Query has a GAP (what's missing)
        - Good answer FILLS that gap
        - The SUBJECT of the answer should match the query subject
        """
        gap_type, fill_words = self._detect_question_type(query)
        
        if gap_type is None:
            return 0.0
        
        candidate_words = set(self._tokenize(candidate))
        
        # How many fill words does the candidate have?
        fill_overlap = len(candidate_words & fill_words)
        fill_score = fill_overlap / max(len(fill_words), 1)
        
        # CRITICAL: Check if the SUBJECT matches
        # "Who is Captain Ahab?" → subject is "Ahab"
        # "Captain Ahab is..." → subject is "Captain Ahab" ✓
        # "The Pequod is..." → subject is "Pequod" ✗
        query_subject = self._extract_subject(query)
        candidate_subject = self._extract_subject(candidate)
        
        # Boost if subjects overlap
        subject_overlap = len(query_subject & candidate_subject)
        if subject_overlap > 0:
            fill_score *= 1.5  # Boost for matching subject
        
        return min(fill_score, 1.0)  # Cap at 1.0
    
    def _compute_content_similarity(self, query: str, candidate: str) -> float:
        """Compute content similarity (what they're both about)."""
        query_pos = self._encode_content(query)
        candidate_pos = self._encode_content(candidate)
        
        norm_q = np.linalg.norm(query_pos)
        norm_c = np.linalg.norm(candidate_pos)
        
        if norm_q < 1e-8 or norm_c < 1e-8:
            return 0.0
        
        return np.dot(query_pos, candidate_pos) / (norm_q * norm_c)
    
    def _compute_holographic_score(self, query: str, candidate: str) -> Tuple[float, Dict]:
        """
        Compute holographic match score.
        
        Score = Content_overlap × Gap_fill
        
        Both must be present:
        - Content overlap ensures we're talking about the same thing
        - Gap fill ensures the answer provides what's missing
        """
        content_sim = self._compute_content_similarity(query, candidate)
        gap_fill = self._compute_gap_fill_score(query, candidate)
        
        # Also check direct word overlap for content words
        query_content = self._extract_content_words(query)
        candidate_content = self._extract_content_words(candidate)
        word_overlap = len(query_content & candidate_content) / max(len(query_content), 1)
        
        # Combined score: need BOTH content match AND gap fill
        # The holographic insight: gap_fill should MULTIPLY, not just add
        # An answer that doesn't fill the gap is wrong regardless of content overlap
        
        # Base content score
        content_score = 0.5 * content_sim + 0.5 * word_overlap
        
        # Gap fill acts as a multiplier - no fill = heavily penalized
        # gap_fill ranges 0-1, so (0.5 + gap_fill) ranges 0.5-1.5
        gap_multiplier = 0.3 + 0.7 * gap_fill  # ranges 0.3-1.0
        
        holographic_score = content_score * gap_multiplier
        
        details = {
            'content_sim': content_sim,
            'gap_fill': gap_fill,
            'word_overlap': word_overlap,
            'gap_multiplier': gap_multiplier,
            'holographic': holographic_score,
        }
        
        return holographic_score, details
    
    def store(self, text: str, fact_id: str = None):
        """Store a fact."""
        if fact_id is None:
            fact_id = f"fact_{len(self.facts)}"
        self.facts.append((text, fact_id))
    
    def query(self, question: str, top_k: int = 3) -> List[Tuple[str, str, float, Dict]]:
        """
        Query for answers that fill the question's gap.
        
        Returns: List of (text, id, score, details)
        """
        results = []
        
        gap_type, fill_words = self._detect_question_type(question)
        
        for text, fact_id in self.facts:
            score, details = self._compute_holographic_score(question, text)
            details['gap_type'] = gap_type
            details['fill_words'] = fill_words
            results.append((text, fact_id, score, details))
        
        # Sort by holographic score
        results.sort(key=lambda x: -x[2])
        
        return results[:top_k]


def main():
    print("=" * 70)
    print("HOLOGRAPHIC Q&A SYSTEM")
    print("Matching based on GAP-FILLING, not just word overlap")
    print("=" * 70)
    
    qa = HolographicQA(dim=32)
    
    # Store some Moby Dick facts
    facts = [
        # Descriptions (fill WHO gaps)
        "Captain Ahab is the monomaniacal captain of the Pequod, obsessed with hunting the white whale.",
        "Ishmael is the narrator of the story, a young sailor who joins the Pequod's crew.",
        "Queequeg is a harpooner from the South Pacific, covered in tattoos and carrying a tomahawk.",
        "Starbuck is the first mate of the Pequod, a cautious and religious man from Nantucket.",
        
        # Definitions (fill WHAT gaps)
        "Moby Dick is a giant white sperm whale that bit off Captain Ahab's leg.",
        "The Pequod is the whaling ship commanded by Captain Ahab.",
        "A harpoon is a long spear used to kill whales from a small boat.",
        
        # Events (fill WHEN gaps)
        "Ahab lost his leg to Moby Dick on a previous voyage, years before the story begins.",
        "The Pequod sank at the end of the voyage when Moby Dick destroyed it.",
        
        # Locations (fill WHERE gaps)
        "The Pequod sailed from Nantucket, a whaling port in Massachusetts.",
        "The final chase took place in the Pacific Ocean near the equator.",
        
        # Reasons (fill WHY gaps)
        "Ahab hunts Moby Dick because the whale bit off his leg and he seeks revenge.",
        "Ishmael went to sea because he felt depressed and wanted adventure.",
        
        # Methods (fill HOW gaps)
        "Whales are hunted by throwing harpoons from small boats launched from the ship.",
        "The crew spots whales by climbing to the mast-head and watching the horizon.",
        
        # Random dialogue (should NOT match well)
        "Have ye clapped eye on Captain Ahab?",
        "Aye, aye, sir! The harpoons are ready!",
        "Thar she blows! A whale! A whale!",
    ]
    
    for fact in facts:
        qa.store(fact)
    
    print(f"\nStored {len(qa.facts)} facts")
    
    # Test queries
    print("\n" + "=" * 70)
    print("QUERY TESTS")
    print("=" * 70)
    
    queries = [
        "Who is Captain Ahab?",
        "What is Moby Dick?",
        "Why does Ahab hunt the whale?",
        "Where did the Pequod sail from?",
        "How do they hunt whales?",
        "Tell me about Queequeg",
        "When did Ahab lose his leg?",
    ]
    
    for question in queries:
        print(f"\n{'─' * 70}")
        print(f"Q: {question}")
        
        gap_type, fill_words = qa._detect_question_type(question)
        print(f"   Gap type: {gap_type}")
        print(f"   Looking for: {fill_words}")
        
        results = qa.query(question, top_k=3)
        
        print(f"\n   Top matches:")
        for i, (text, fid, score, details) in enumerate(results):
            preview = text[:80] + "..." if len(text) > 80 else text
            print(f"   [{i+1}] Score: {score:.3f} (content={details['content_sim']:.2f}, gap_fill={details['gap_fill']:.2f})")
            print(f"       {preview}")


if __name__ == "__main__":
    main()
