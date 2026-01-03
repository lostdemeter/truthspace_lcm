#!/usr/bin/env python3
"""
Moby Dick Holographic Q&A Demo

An interactive chatbot that demonstrates the holographic Q&A matching system
described in the paper "Holographic Question-Answer Matching: Gaps as Information".

This demo:
1. Downloads Moby Dick from Project Gutenberg
2. Extracts meaningful sentences and facts
3. Allows interactive Q&A using holographic gap-filling

Usage:
    python moby_dick_demo.py
"""

import sys
import os
import re
import urllib.request
from typing import Dict, List, Tuple, Set, Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHI = (1 + np.sqrt(5)) / 2

# ============================================================================
# GAP TAXONOMY - What kind of information is the question asking for?
# ============================================================================

QUESTION_GAPS = {
    'IDENTITY': {
        'patterns': ['who is', 'who was', 'who are', 'who were', 'tell me about', 'describe'],
        'fill_words': {'is', 'was', 'named', 'called', 'known', 'captain', 'man', 'woman', 'person', 'sailor', 'harpooner'},
    },
    'EVENT': {
        'patterns': ['what happened', 'what occurs', 'what took place', 'did what'],
        'fill_words': {'sank', 'destroyed', 'killed', 'died', 'survived', 'happened', 'end', 'final', 'drowned', 'lost'},
    },
    'DEFINITION': {
        'patterns': ['what is', 'what was', 'what are', 'what does', 'define', 'meaning of'],
        'fill_words': {'is', 'means', 'refers', 'defined', 'called', 'type', 'kind', 'whale', 'ship'},
    },
    'TIME': {
        'patterns': ['when did', 'when was', 'when is', 'what year', 'what date', 'what time', 'how long'],
        'fill_words': {'year', 'date', 'time', 'day', 'month', 'century', 'ago', 'in', 'on', 'at', 'before', 'after', 'during'},
    },
    'LOCATION': {
        'patterns': ['where is', 'where was', 'where did', 'what place', 'location of', 'from where'],
        'fill_words': {'in', 'at', 'near', 'city', 'country', 'place', 'located', 'found', 'sea', 'ocean', 'port', 'nantucket'},
    },
    'REASON': {
        'patterns': ['why did', 'why does', 'why is', 'why was', 'reason', 'cause of', 'what made'],
        'fill_words': {'because', 'since', 'due', 'reason', 'cause', 'therefore', 'so', 'revenge', 'obsessed'},
    },
    'METHOD': {
        'patterns': ['how did', 'how does', 'how do', 'how to', 'how is', 'how was', 'how are', 'method', 'way to', 'process'],
        'fill_words': {'by', 'through', 'using', 'method', 'way', 'process', 'step', 'harpoon', 'boat', 'throwing', 'from'},
    },
}

# ============================================================================
# CONTENT SEEDS - Semantic clusters for Moby Dick domain
# ============================================================================

MOBY_DICK_SEEDS = {
    # Characters
    'AHAB': ['ahab', 'captain', 'commander', 'leg', 'ivory', 'monomaniac', 'obsessed'],
    'ISHMAEL': ['ishmael', 'narrator', 'schoolmaster', 'sailor'],
    'QUEEQUEG': ['queequeg', 'harpooner', 'savage', 'cannibal', 'coffin', 'tattoo', 'tomahawk'],
    'STARBUCK': ['starbuck', 'mate', 'first', 'nantucket', 'quaker', 'prudent'],
    'STUBB': ['stubb', 'second', 'pipe', 'jolly', 'easygoing'],
    'FLASK': ['flask', 'third', 'short', 'pugnacious'],
    'TASHTEGO': ['tashtego', 'indian', 'gay', 'head'],
    'DAGGOO': ['daggoo', 'african', 'giant', 'tall'],
    'FEDALLAH': ['fedallah', 'parsee', 'prophet', 'mysterious', 'shadow'],
    'PIP': ['pip', 'boy', 'cabin', 'mad', 'tambourine'],
    
    # The whale
    'WHALE': ['whale', 'moby', 'dick', 'leviathan', 'sperm', 'white', 'monster', 'creature'],
    'WHALING': ['whaling', 'hunt', 'chase', 'pursue', 'harpoon', 'kill', 'blubber', 'oil'],
    
    # Ship and sea
    'SHIP': ['ship', 'pequod', 'vessel', 'boat', 'deck', 'mast', 'sail', 'hull'],
    'SEA': ['sea', 'ocean', 'water', 'wave', 'deep', 'voyage', 'pacific', 'atlantic'],
    
    # Themes
    'DEATH': ['death', 'die', 'dead', 'doom', 'fate', 'destruction', 'end', 'sank', 'drowned'],
    'OBSESSION': ['obsession', 'madness', 'revenge', 'vengeance', 'monomaniac', 'hate'],
    'NATURE': ['nature', 'god', 'divine', 'creation', 'universe', 'power'],
    
    # Places
    'NANTUCKET': ['nantucket', 'massachusetts', 'new', 'bedford', 'england', 'america'],
    'PACIFIC': ['pacific', 'equator', 'japan', 'line'],
}


class HolographicMobyDick:
    """
    Holographic Q&A system specialized for Moby Dick.
    
    Uses gap-filling to match questions to answers based on intent,
    not just word overlap.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.facts: List[Tuple[str, str]] = []  # (text, source)
        self.word_positions: Dict[str, np.ndarray] = {}
        self.seed_positions: Dict[str, np.ndarray] = {}
        
        self._init_seeds()
    
    def _init_seeds(self):
        """Initialize seed positions."""
        for seed_name in MOBY_DICK_SEEDS.keys():
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
        
        # Check seeds
        for seed_name, seed_words in MOBY_DICK_SEEDS.items():
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
    
    def _detect_gap_type(self, text: str) -> Tuple[Optional[str], Set[str]]:
        """Detect question type and expected fill words."""
        text_lower = text.lower()
        
        for gap_type, info in QUESTION_GAPS.items():
            for pattern in info['patterns']:
                if pattern in text_lower:
                    return gap_type, info['fill_words']
        
        return None, set()
    
    def _extract_content_words(self, text: str) -> Set[str]:
        """Extract content words (remove stop words)."""
        words = set(self._tokenize(text))
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'to', 'of', 'and', 'in', 'on', 'at', 'for', 'with', 'by', 'from',
            'what', 'when', 'where', 'who', 'how', 'why', 'which', 'that', 'this',
            'do', 'does', 'did', 'i', 'you', 'it', 'he', 'she', 'they', 'we',
            'me', 'my', 'about', 'tell', 'can', 'could', 'would', 'should',
            'have', 'has', 'had', 'will', 'shall', 'may', 'might', 'must',
            'his', 'her', 'its', 'their', 'our', 'your', 'him', 'them', 'us',
        }
        return words - stop_words
    
    def _extract_subject(self, text: str) -> Set[str]:
        """Extract the subject of a sentence."""
        words = self._tokenize(text)
        question_words = {'who', 'what', 'when', 'where', 'why', 'how', 'which'}
        
        # For questions: subject after verb
        if words and words[0] in question_words:
            for i, word in enumerate(words):
                if word in ['is', 'was', 'are', 'were', 'does', 'did']:
                    subject_words = set(words[i+1:])
                    return subject_words - {'the', 'a', 'an'}
        
        # For statements: subject before verb
        for i, word in enumerate(words):
            if word in ['is', 'was', 'are', 'were', 'has', 'had', 'does', 'did']:
                subject_words = set(words[:i])
                return subject_words - {'the', 'a', 'an', 'who', 'what', 'when', 'where', 'why', 'how'}
        
        return set()
    
    def _encode_content(self, text: str) -> np.ndarray:
        """Encode text as average of content word positions."""
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
        
        details = {
            'content_sim': content_sim,
            'word_overlap': word_overlap,
            'gap_type': gap_type,
            'gap_fill': gap_fill,
            'gap_multiplier': gap_multiplier,
        }
        
        return holographic_score, details
    
    def store(self, text: str, source: str = "unknown"):
        """Store a fact."""
        self.facts.append((text, source))
    
    def query(self, question: str, top_k: int = 3) -> List[Tuple[str, str, float, Dict]]:
        """Query for answers."""
        results = []
        
        for text, source in self.facts:
            score, details = self._compute_holographic_score(question, text)
            results.append((text, source, score, details))
        
        results.sort(key=lambda x: -x[2])
        return results[:top_k]
    
    def answer(self, question: str) -> Tuple[str, float, Dict]:
        """Get the best answer to a question."""
        results = self.query(question, top_k=1)
        if results:
            text, source, score, details = results[0]
            return text, score, details
        return "I don't know.", 0.0, {}


def download_moby_dick() -> str:
    """Download Moby Dick from Project Gutenberg."""
    url = "https://www.gutenberg.org/files/2701/2701-0.txt"
    cache_path = "/tmp/moby_dick.txt"
    
    if os.path.exists(cache_path):
        print("  Loading from cache...")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    print("  Downloading from Project Gutenberg...")
    with urllib.request.urlopen(url) as response:
        text = response.read().decode('utf-8')
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return text


def extract_facts(text: str) -> List[Tuple[str, str]]:
    """
    Extract meaningful facts from Moby Dick.
    
    Returns list of (fact_text, chapter_source) tuples.
    """
    # Find content boundaries
    start_marker = "CHAPTER 1. Loomings."
    end_marker = "*** END OF THE PROJECT GUTENBERG"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1:
        text = text[start_idx:]
    if end_idx != -1:
        text = text[:end_idx]
    
    facts = []
    
    # Extract chapter-by-chapter
    chapter_pattern = r'CHAPTER\s+(\d+)\.\s+([^\n]+)'
    matches = list(re.finditer(chapter_pattern, text))
    
    for i, match in enumerate(matches):
        chapter_num = match.group(1)
        chapter_title = match.group(2).strip()
        
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        chapter_content = text[start:end]
        chapter_source = f"Chapter {chapter_num}: {chapter_title}"
        
        # Extract sentences that look like facts (contain "is", "was", descriptive)
        sentences = re.split(r'(?<=[.!?])\s+', chapter_content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            sentence = re.sub(r'\s+', ' ', sentence)
            
            # Filter for good fact candidates
            if len(sentence) < 30 or len(sentence) > 400:
                continue
            
            # Look for descriptive sentences
            words = sentence.lower().split()
            
            # Must contain a verb that indicates description/fact
            fact_indicators = {'is', 'was', 'are', 'were', 'has', 'had', 'called', 'named'}
            if not any(ind in words for ind in fact_indicators):
                continue
            
            # Skip dialogue (starts with quote)
            if sentence.startswith('"') or sentence.startswith("'"):
                continue
            
            # Skip questions
            if sentence.endswith('?'):
                continue
            
            facts.append((sentence, chapter_source))
    
    return facts


def add_curated_facts(qa: HolographicMobyDick):
    """Add hand-curated facts for better coverage."""
    curated = [
        # Characters
        ("Captain Ahab is the monomaniacal captain of the Pequod, a whaling ship, who is obsessed with hunting the white whale Moby Dick that bit off his leg.", "Curated: Characters"),
        ("Ishmael is the narrator of Moby Dick, a young sailor who joins the crew of the Pequod and is the sole survivor of the voyage.", "Curated: Characters"),
        ("Queequeg is a harpooner from the South Pacific island of Rokovoko, covered in tattoos and carrying a tomahawk, who becomes Ishmael's close friend.", "Curated: Characters"),
        ("Starbuck is the first mate of the Pequod, a cautious, religious Quaker from Nantucket who opposes Ahab's obsessive quest.", "Curated: Characters"),
        ("Stubb is the second mate of the Pequod, known for his easygoing nature and his ever-present pipe.", "Curated: Characters"),
        ("Flask is the third mate of the Pequod, a short, pugnacious man from Martha's Vineyard.", "Curated: Characters"),
        ("Fedallah is a mysterious Parsee prophet who serves as Ahab's personal harpooner and makes dark prophecies about Ahab's fate.", "Curated: Characters"),
        ("Pip is the young African-American cabin boy who goes mad after being abandoned in the ocean during a whale hunt.", "Curated: Characters"),
        
        # The whale
        ("Moby Dick is a giant white sperm whale that bit off Captain Ahab's leg on a previous voyage, making Ahab obsessed with revenge.", "Curated: The Whale"),
        ("The white whale Moby Dick is famous among whalers for his ferocity and the many ships and men he has destroyed.", "Curated: The Whale"),
        ("Moby Dick has a distinctive white color, a wrinkled forehead, and a crooked jaw that makes him recognizable.", "Curated: The Whale"),
        
        # The ship
        ("The Pequod is a whaling ship from Nantucket, commanded by Captain Ahab, that sets sail to hunt whales in the Pacific Ocean.", "Curated: The Ship"),
        ("The Pequod is decorated with whale bones and teeth, giving it a distinctive appearance.", "Curated: The Ship"),
        ("The Pequod sailed from Nantucket, a famous whaling port in Massachusetts, on Christmas Day.", "Curated: The Ship"),
        
        # Events
        ("Ahab lost his leg to Moby Dick on a previous whaling voyage, which drove him to obsessive revenge.", "Curated: Events"),
        ("The Pequod sank at the end of the novel when Moby Dick destroyed it, killing everyone except Ishmael.", "Curated: Events"),
        ("Ishmael survived the sinking of the Pequod by clinging to Queequeg's coffin, which floated to the surface.", "Curated: Events"),
        ("The final chase of Moby Dick lasted three days before the whale destroyed the Pequod.", "Curated: Events"),
        
        # Themes and reasons
        ("Ahab hunts Moby Dick because the whale bit off his leg and he seeks revenge, seeing the whale as the embodiment of evil.", "Curated: Themes"),
        ("Ishmael went to sea because he felt depressed and restless, seeing the ocean as a way to escape his melancholy.", "Curated: Themes"),
        ("Moby Dick represents the unknowable forces of nature and the limits of human understanding and control.", "Curated: Themes"),
        
        # Whaling
        ("Whales are hunted by throwing harpoons from small boats called whaleboats, launched from the main ship.", "Curated: Whaling"),
        ("Sperm whales are hunted for their oil, which was used for lamps and lubrication in the 19th century.", "Curated: Whaling"),
        ("The crew spots whales by climbing to the mast-head and watching the horizon for spouts.", "Curated: Whaling"),
        
        # Places
        ("Nantucket is a small island off the coast of Massachusetts that was the center of the American whaling industry.", "Curated: Places"),
        ("The Pequod sailed through the Atlantic Ocean, around Cape Horn, and into the Pacific Ocean in pursuit of whales.", "Curated: Places"),
        ("The final confrontation with Moby Dick took place in the Pacific Ocean near the equator.", "Curated: Places"),
        
        # Famous quotes context
        ("The novel begins with the famous line 'Call me Ishmael,' introducing the narrator.", "Curated: Famous Lines"),
        ("Ahab's famous speech on the quarter-deck reveals his obsessive hatred of the white whale.", "Curated: Famous Lines"),
    ]
    
    for fact, source in curated:
        qa.store(fact, source)


def main():
    print("=" * 70)
    print("  MOBY DICK HOLOGRAPHIC Q&A DEMO")
    print("  Demonstrating gap-filling semantic matching")
    print("=" * 70)
    
    # Initialize
    qa = HolographicMobyDick(dim=64)
    
    # Download and process
    print("\n[1] Downloading Moby Dick...")
    text = download_moby_dick()
    print(f"    Downloaded {len(text):,} characters")
    
    print("\n[2] Extracting facts...")
    facts = extract_facts(text)
    print(f"    Extracted {len(facts):,} fact candidates")
    
    # Store facts (limit to avoid overwhelming)
    print("\n[3] Ingesting facts...")
    
    # Add curated facts FIRST for priority
    add_curated_facts(qa)
    curated_count = len(qa.facts)
    
    # Then add extracted facts
    for fact, source in facts[:500]:  # First 500 facts
        qa.store(fact, source)
    print(f"    Total facts stored: {len(qa.facts):,}")
    print(f"    Vocabulary size: {len(qa.word_positions):,} words")
    
    # Demo queries
    print("\n" + "=" * 70)
    print("  DEMO QUERIES")
    print("=" * 70)
    
    demo_queries = [
        "Who is Captain Ahab?",
        "What is Moby Dick?",
        "Why does Ahab hunt the whale?",
        "Where did the Pequod sail from?",
        "How do they hunt whales?",
        "Tell me about Queequeg",
        "What happened to the Pequod?",
        "Who is the narrator?",
    ]
    
    for query in demo_queries:
        print(f"\n{'─' * 70}")
        print(f"Q: {query}")
        
        answer, score, details = qa.answer(query)
        
        gap_type = details.get('gap_type', 'None')
        gap_fill = details.get('gap_fill', 0)
        
        print(f"   [Gap: {gap_type}, Fill: {gap_fill:.2f}, Score: {score:.3f}]")
        
        # Truncate long answers
        if len(answer) > 200:
            answer = answer[:200] + "..."
        print(f"A: {answer}")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("  INTERACTIVE MODE")
    print("  Ask questions about Moby Dick!")
    print("  Type 'quit' to exit, 'top3' to see top 3 matches")
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
        
        if user_input.lower() == 'help':
            print("""
  Example questions:
    - Who is Captain Ahab?
    - What is Moby Dick?
    - Why does Ahab hunt the whale?
    - Where did the Pequod sail from?
    - How do they hunt whales?
    - Tell me about Queequeg
    - What happened to the Pequod?
    - Who survived the voyage?
    - When did the Pequod sail?
    - What is the Pequod?
            """)
            continue
        
        if show_top3:
            results = qa.query(user_input, top_k=3)
            print(f"\n  Top 3 matches:")
            for i, (text, source, score, details) in enumerate(results):
                preview = text[:150] + "..." if len(text) > 150 else text
                print(f"\n  [{i+1}] Score: {score:.3f} ({source})")
                print(f"      {preview}")
        else:
            answer, score, details = qa.answer(user_input)
            
            gap_type = details.get('gap_type', 'None')
            gap_fill = details.get('gap_fill', 0)
            
            print(f"   [Gap: {gap_type}, Fill: {gap_fill:.2f}, Score: {score:.3f}]")
            
            if len(answer) > 300:
                answer = answer[:300] + "..."
            print(f"Ahab: {answer}")


if __name__ == "__main__":
    main()
