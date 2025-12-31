#!/usr/bin/env python3
"""
Book Trainer - Continuous Training on Literature

Reads a book line by line, extracts concepts, queries LLM for unknowns,
and continuously trains the emergent structure.

This is an experiment to see how much emergent structure can be built
from a single book (e.g., Moby Dick).
"""

import json
import numpy as np
import requests
import time
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.fully_emergent_chains import FullyEmergentSemanticChain
from experiments.segmented_rebalance import SegmentedStructure


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2:latest"

# Common words to skip (will also use emergent stopwords)
SKIP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
    'and', 'or', 'but', 'if', 'then', 'so', 'than', 'too', 'very', 'just',
    'not', 'no', 'yes', 'all', 'any', 'some', 'every', 'each', 'both',
    'few', 'more', 'most', 'other', 'such', 'only', 'own', 'same', 'that',
    'this', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose',
    'when', 'where', 'why', 'how', 'i', 'me', 'my', 'mine', 'we', 'us',
    'our', 'ours', 'you', 'your', 'yours', 'he', 'him', 'his', 'she',
    'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs',
    'there', 'here', 'now', 'then', 'also', 'still', 'already', 'yet',
    'again', 'ever', 'never', 'always', 'often', 'sometimes', 'usually',
    'about', 'after', 'before', 'between', 'under', 'over', 'above',
    'below', 'up', 'down', 'out', 'off', 'away', 'back', 'around',
    'upon', 'along', 'across', 'against', 'among', 'within', 'without',
    'during', 'until', 'while', 'since', 'because', 'although', 'though',
    'unless', 'whether', 'once', 'twice', 'much', 'many', 'little',
    'less', 'least', 'enough', 'quite', 'rather', 'almost', 'even',
    'well', 'far', 'near', 'long', 'short', 'high', 'low', 'old', 'new',
    'good', 'bad', 'great', 'small', 'large', 'big', 'first', 'last',
    'next', 'right', 'left', 'said', 'says', 'say', 'told', 'tell',
    'asked', 'ask', 'went', 'go', 'goes', 'going', 'gone', 'came',
    'come', 'comes', 'coming', 'made', 'make', 'makes', 'making',
    'took', 'take', 'takes', 'taking', 'got', 'get', 'gets', 'getting',
    'gave', 'give', 'gives', 'giving', 'put', 'puts', 'putting',
    'let', 'lets', 'letting', 'see', 'sees', 'seeing', 'saw', 'seen',
    'know', 'knows', 'knowing', 'knew', 'known', 'think', 'thinks',
    'thinking', 'thought', 'want', 'wants', 'wanting', 'wanted',
    'like', 'likes', 'liking', 'liked', 'need', 'needs', 'needing',
    'needed', 'seem', 'seems', 'seeming', 'seemed', 'look', 'looks',
    'looking', 'looked', 'use', 'uses', 'using', 'used', 'find',
    'finds', 'finding', 'found', 'keep', 'keeps', 'keeping', 'kept',
    'begin', 'begins', 'beginning', 'began', 'begun', 'end', 'ends',
    'ending', 'ended', 'turn', 'turns', 'turning', 'turned',
    'show', 'shows', 'showing', 'showed', 'shown', 'try', 'tries',
    'trying', 'tried', 'leave', 'leaves', 'leaving', 'left',
    'call', 'calls', 'calling', 'called', 'must', 'might', 'may',
    'can', 'could', 'would', 'should', 'shall', 'will', 'being',
    'been', 'having', 'doing', 'done', 'did', 'does', 'had', 'has',
}


@dataclass
class TrainingStats:
    """Statistics from training."""
    lines_processed: int = 0
    sentences_extracted: int = 0
    concepts_found: int = 0
    concepts_new: int = 0
    concepts_injected: int = 0
    llm_queries: int = 0
    rebalances: int = 0
    anchors_discovered: int = 0
    start_time: float = field(default_factory=time.time)
    
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    def rate(self) -> float:
        elapsed = self.elapsed()
        if elapsed > 0:
            return self.lines_processed / elapsed
        return 0.0


class BookTrainer:
    """
    Trains emergent structure from a book, line by line.
    
    Process:
    1. Read line
    2. Extract potential concepts (proper nouns, significant words)
    3. For unknown concepts, query LLM for behavioral description
    4. Inject into structure
    5. Periodically rebalance
    """
    
    def __init__(self):
        self.semantic = FullyEmergentSemanticChain()
        self.structure: Optional[SegmentedStructure] = None
        self.stats = TrainingStats()
        
        # Track concepts
        self.known_concepts: Set[str] = set()
        self.concept_counts: Counter = Counter()
        self.pending_concepts: List[str] = []
        
        # Book metadata
        self.book_title = ""
        self.current_chapter = ""
        
    def _call_llm(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """Call Ollama API."""
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.4}
                },
                timeout=30
            )
            if response.status_code == 200:
                self.stats.llm_queries += 1
                return response.json().get("response", "").strip()
        except Exception as e:
            pass
        return None
    
    def initialize(self, seed_corpus_path: Optional[str] = None):
        """Initialize with optional seed corpus."""
        if seed_corpus_path and Path(seed_corpus_path).exists():
            print(f"Loading seed corpus: {seed_corpus_path}")
            count = self.semantic.ingest_corpus(seed_corpus_path)
            print(f"  Loaded {count} seed items")
        
        # Initial dimension learning
        if self.semantic.items:
            self.semantic.learn_dimensions()
            self.known_concepts = set(self.semantic.groups)
        
        # Create structure manager
        self.structure = SegmentedStructure(self.semantic)
        
        print(f"Initialized with {len(self.known_concepts)} known concepts")
    
    def _extract_concepts_from_line(self, line: str) -> List[str]:
        """Extract potential concepts from a line of text."""
        concepts = []
        
        # Clean line
        line = line.strip()
        if not line or len(line) < 10:
            return []
        
        # Find capitalized words (potential proper nouns)
        words = re.findall(r'\b[A-Z][a-z]+\b', line)
        for word in words:
            word_lower = word.lower()
            if word_lower not in SKIP_WORDS and len(word_lower) >= 3:
                concepts.append(word_lower)
        
        # Find quoted terms
        quoted = re.findall(r'"([^"]+)"', line)
        for term in quoted:
            term_lower = term.lower().strip()
            if len(term_lower) >= 3 and ' ' not in term_lower:
                concepts.append(term_lower)
        
        return list(set(concepts))
    
    def _is_known_concept(self, concept: str) -> bool:
        """Check if concept is already known."""
        concept = concept.lower()
        return concept in self.known_concepts or concept in self.semantic.groups
    
    def _inject_concept(self, concept: str, context_line: str) -> bool:
        """Inject a new concept by querying LLM for behavioral description."""
        concept_title = concept.replace('_', ' ').title()
        
        # Query LLM for behavioral sentences
        prompt = f"""Generate 3 behavioral sentences about "{concept_title}" based on this context:
"{context_line[:200]}"

Rules:
1. Start each sentence with "{concept_title}"
2. Second word should be a verb (action)
3. Keep sentences 8-15 words
4. Describe characteristic behaviors or traits

Generate 3 sentences:"""

        response = self._call_llm(prompt, max_tokens=250)
        if not response:
            return False
        
        injected = 0
        for line in response.strip().split('\n'):
            line = line.strip().lstrip('0123456789.-) ')
            if len(line) > 15 and line.lower().startswith(concept_title.lower()):
                self.semantic.ingest_item({
                    'text': line,
                    'agent': concept.lower(),
                    'source': f'book_training:{self.book_title}',
                    'chapter': self.current_chapter,
                })
                injected += 1
        
        if injected > 0:
            self.known_concepts.add(concept.lower())
            self.stats.concepts_injected += 1
            return True
        
        return False
    
    def process_line(self, line: str) -> Dict:
        """Process a single line from the book."""
        self.stats.lines_processed += 1
        
        # Check for chapter markers
        if line.strip().startswith('CHAPTER'):
            self.current_chapter = line.strip()[:50]
            return {'type': 'chapter', 'chapter': self.current_chapter}
        
        # Extract concepts
        concepts = self._extract_concepts_from_line(line)
        
        result = {
            'type': 'line',
            'concepts_found': len(concepts),
            'new_concepts': [],
            'injected': [],
        }
        
        for concept in concepts:
            self.concept_counts[concept] += 1
            self.stats.concepts_found += 1
            
            if not self._is_known_concept(concept):
                result['new_concepts'].append(concept)
                self.stats.concepts_new += 1
                
                # Add to pending if seen multiple times
                if self.concept_counts[concept] >= 2:
                    if concept not in self.pending_concepts:
                        self.pending_concepts.append(concept)
        
        # Also ingest the line itself if it has a clear subject
        if concepts and len(line) > 20:
            # Use first concept as agent
            self.semantic.ingest_item({
                'text': line.strip(),
                'agent': concepts[0],
                'source': f'book:{self.book_title}',
            })
            self.stats.sentences_extracted += 1
        
        return result
    
    def process_pending_concepts(self, max_concepts: int = 5):
        """Process pending concepts by querying LLM."""
        if not self.pending_concepts:
            return
        
        # Sort by frequency (most common first)
        self.pending_concepts.sort(key=lambda c: -self.concept_counts[c])
        
        processed = 0
        for concept in self.pending_concepts[:max_concepts]:
            # Find a context line for this concept
            context = f"The concept '{concept}' appears in {self.book_title}"
            
            if self._inject_concept(concept, context):
                processed += 1
                print(f"    Injected: {concept} (seen {self.concept_counts[concept]}x)")
            
            time.sleep(0.2)  # Rate limit
        
        # Remove processed concepts
        self.pending_concepts = self.pending_concepts[max_concepts:]
    
    def rebalance(self):
        """Perform segmented rebalancing."""
        if not self.structure:
            return
        
        # Rediscover dimensions with new data
        self.semantic.learn_dimensions()
        
        # Update structure
        self.structure.discover_segments()
        self.structure.rebalance_all_segments()
        
        self.stats.rebalances += 1
    
    def discover_anchors(self):
        """Discover anchor points in current structure."""
        if not self.structure:
            return
        
        self.structure.discover_anchors(min_confidence=0.7)
        self.stats.anchors_discovered = len(self.structure.anchors)
    
    def train_on_text(self, text: str, title: str = "Unknown", 
                      max_lines: int = None,
                      rebalance_every: int = 100,
                      inject_every: int = 50,
                      progress_every: int = 100):
        """
        Train on a text, line by line.
        
        Args:
            text: The full text to process
            title: Book title
            max_lines: Maximum lines to process (None = all)
            rebalance_every: Rebalance every N lines
            inject_every: Process pending concepts every N lines
            progress_every: Print progress every N lines
        """
        self.book_title = title
        lines = text.split('\n')
        
        if max_lines:
            lines = lines[:max_lines]
        
        print(f"\n{'='*60}")
        print(f"TRAINING ON: {title}")
        print(f"{'='*60}")
        print(f"Total lines: {len(lines)}")
        print(f"Rebalance every: {rebalance_every} lines")
        print(f"Inject every: {inject_every} lines")
        
        for i, line in enumerate(lines):
            # Process line
            result = self.process_line(line)
            
            # Periodic injection
            if (i + 1) % inject_every == 0 and self.pending_concepts:
                print(f"\n  [Line {i+1}] Processing {len(self.pending_concepts)} pending concepts...")
                self.process_pending_concepts(max_concepts=3)
            
            # Periodic rebalancing
            if (i + 1) % rebalance_every == 0:
                print(f"\n  [Line {i+1}] Rebalancing...")
                self.rebalance()
            
            # Progress report
            if (i + 1) % progress_every == 0:
                self._print_progress(i + 1, len(lines))
        
        # Final processing
        print(f"\n{'─'*60}")
        print("FINAL PROCESSING")
        print(f"{'─'*60}")
        
        # Process remaining pending concepts
        while self.pending_concepts:
            print(f"Processing {len(self.pending_concepts)} remaining concepts...")
            self.process_pending_concepts(max_concepts=5)
            time.sleep(0.3)
        
        # Final rebalance
        print("Final rebalance...")
        self.rebalance()
        
        # Discover anchors
        print("Discovering anchors...")
        self.discover_anchors()
    
    def _print_progress(self, current: int, total: int):
        """Print progress report."""
        pct = current / total * 100
        rate = self.stats.rate()
        
        print(f"\n  Progress: {current}/{total} ({pct:.1f}%)")
        print(f"    Rate: {rate:.1f} lines/sec")
        print(f"    Concepts: {len(self.known_concepts)} known, {self.stats.concepts_injected} injected")
        print(f"    Sentences: {self.stats.sentences_extracted}")
        print(f"    LLM queries: {self.stats.llm_queries}")
        print(f"    Pending: {len(self.pending_concepts)}")
    
    def save_state(self, path: str):
        """Save trained state to file."""
        state = {
            'known_concepts': list(self.known_concepts),
            'concept_counts': dict(self.concept_counts),
            'book_title': self.book_title,
            'stats': {
                'lines_processed': self.stats.lines_processed,
                'sentences_extracted': self.stats.sentences_extracted,
                'concepts_found': self.stats.concepts_found,
                'concepts_new': self.stats.concepts_new,
                'concepts_injected': self.stats.concepts_injected,
                'llm_queries': self.stats.llm_queries,
                'rebalances': self.stats.rebalances,
            }
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Saved state to {path}")
    
    def print_summary(self):
        """Print training summary."""
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        
        print(f"\nProcessing:")
        print(f"  Lines processed: {self.stats.lines_processed}")
        print(f"  Sentences extracted: {self.stats.sentences_extracted}")
        print(f"  Time elapsed: {self.stats.elapsed():.1f}s")
        print(f"  Rate: {self.stats.rate():.1f} lines/sec")
        
        print(f"\nConcepts:")
        print(f"  Total found: {self.stats.concepts_found}")
        print(f"  New concepts: {self.stats.concepts_new}")
        print(f"  Injected: {self.stats.concepts_injected}")
        print(f"  Final known: {len(self.known_concepts)}")
        
        print(f"\nStructure:")
        print(f"  Dimensions: {len(self.semantic.dimensions)}")
        print(f"  Rebalances: {self.stats.rebalances}")
        print(f"  Anchors: {self.stats.anchors_discovered}")
        print(f"  Segments: {len(self.structure.segments) if self.structure else 0}")
        
        print(f"\nLLM:")
        print(f"  Queries: {self.stats.llm_queries}")
        
        # Top concepts by frequency
        print(f"\nTop concepts by frequency:")
        for concept, count in self.concept_counts.most_common(15):
            known = "✓" if concept in self.known_concepts else "○"
            print(f"  {known} {concept}: {count}")
        
        # Show some anchors
        if self.structure and self.structure.anchors:
            print(f"\nDiscovered anchors:")
            for anchor in self.structure.anchors[:10]:
                print(f"  {anchor.concept_a} ↔ {anchor.concept_b} (conf={anchor.confidence:.2f})")
    
    def query(self, question: str) -> str:
        """Query the trained structure."""
        # Extract concepts from question
        q_lower = question.lower()
        found = [c for c in self.known_concepts if c in q_lower]
        
        if not found:
            return f"No known concepts found. I know: {', '.join(list(self.known_concepts)[:10])}..."
        
        concept = found[0]
        
        # Get info
        traits = self.semantic.describe_traits(concept)
        similar = self.semantic.find_similar(concept, k=3)
        opposite = self.semantic.find_opposite(concept)
        
        parts = [f"About {concept.title()}:"]
        
        if traits:
            parts.append(f"  Traits: {', '.join(traits)}")
        if similar:
            parts.append(f"  Similar to: {', '.join([s[0] for s in similar])}")
        if opposite:
            parts.append(f"  Opposite: {opposite[0]}")
        
        return '\n'.join(parts)


def fetch_moby_dick() -> str:
    """Fetch Moby Dick from Project Gutenberg."""
    url = "https://www.gutenberg.org/files/2701/2701-0.txt"
    print(f"Fetching Moby Dick from {url}...")
    
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        text = response.text
        
        # Find start of actual content
        start_marker = "CHAPTER 1. Loomings."
        start_idx = text.find(start_marker)
        if start_idx > 0:
            text = text[start_idx:]
        
        # Find end
        end_marker = "*** END OF THE PROJECT GUTENBERG"
        end_idx = text.find(end_marker)
        if end_idx > 0:
            text = text[:end_idx]
        
        print(f"  Fetched {len(text)} characters")
        return text
    
    return ""


def main():
    """Main entry point."""
    print("=" * 70)
    print("BOOK TRAINER - Continuous Learning from Literature")
    print("=" * 70)
    
    # Check Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            print("Ollama not running!")
            return
        print("Ollama is running")
    except:
        print("Ollama not available!")
        return
    
    # Create trainer
    trainer = BookTrainer()
    
    # Initialize with seed corpus (optional)
    base = Path(__file__).parent.parent
    seed_path = base / "truthspace_lcm" / "gears" / "corpus" / "corpus_llm_live.json"
    trainer.initialize(str(seed_path) if seed_path.exists() else None)
    
    # Fetch Moby Dick
    text = fetch_moby_dick()
    if not text:
        print("Failed to fetch book!")
        return
    
    # Train on full book
    trainer.train_on_text(
        text,
        title="Moby Dick",
        max_lines=None,  # Full book
        rebalance_every=500,
        inject_every=200,
        progress_every=1000,
    )
    
    # Print summary
    trainer.print_summary()
    
    # Interactive queries
    print(f"\n{'='*60}")
    print("INTERACTIVE QUERIES")
    print("Type 'quit' to exit")
    print(f"{'='*60}\n")
    
    while True:
        try:
            q = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not q or q.lower() == 'quit':
            break
        
        print(trainer.query(q))
        print()


if __name__ == "__main__":
    main()
