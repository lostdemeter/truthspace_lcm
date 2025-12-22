#!/usr/bin/env python3
"""
Corpus-Wide Pattern Discovery

Runs auto-discovery across multiple literary works to build a 
comprehensive corpus of bootstrap knowledge.

This script:
1. Loads multiple books from Project Gutenberg
2. Runs pattern discovery on each book individually
3. Aggregates patterns across all books
4. Identifies patterns that appear consistently across works
5. Builds a cumulative bootstrap knowledge file
"""

import sys
import os
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.auto_bootstrap import AutoBootstrap
from scripts.ingest_book import clean_gutenberg_text


# Available books
BOOKS = {
    "pride_prejudice": {"path": "/tmp/pride_prejudice.txt", "title": "Pride and Prejudice", "author": "Austen"},
    "alice": {"path": "/tmp/alice.txt", "title": "Alice in Wonderland", "author": "Carroll"},
    "frankenstein": {"path": "/tmp/frankenstein.txt", "title": "Frankenstein", "author": "Shelley"},
    "moby_dick": {"path": "/tmp/moby_dick.txt", "title": "Moby Dick", "author": "Melville"},
    "sherlock": {"path": "/tmp/sherlock.txt", "title": "Sherlock Holmes", "author": "Doyle"},
    "tale_two_cities": {"path": "/tmp/tale_two_cities.txt", "title": "Tale of Two Cities", "author": "Dickens"},
    "prince": {"path": "/tmp/prince.txt", "title": "The Prince", "author": "Machiavelli"},
    "tom_sawyer": {"path": "/tmp/tom_sawyer.txt", "title": "Tom Sawyer", "author": "Twain"},
    "dracula": {"path": "/tmp/dracula.txt", "title": "Dracula", "author": "Stoker"},
    "expectations": {"path": "/tmp/expectations.txt", "title": "Great Expectations", "author": "Dickens"},
}


class CorpusDiscovery:
    """Discover patterns across a corpus of literary works."""
    
    def __init__(self, min_frequency: int = 3, min_books: int = 2):
        """
        Args:
            min_frequency: Minimum frequency within a single book
            min_books: Minimum number of books a pattern must appear in
        """
        self.min_frequency = min_frequency
        self.min_books = min_books
        
        # Results storage
        self.book_results: Dict[str, Dict] = {}
        self.all_patterns: Dict[str, Dict] = {}  # verb -> {books: [], total_freq: int, ...}
        self.corpus_patterns: List[Dict] = []
    
    def load_book(self, book_id: str) -> str:
        """Load and clean a book."""
        if book_id not in BOOKS:
            raise ValueError(f"Unknown book: {book_id}")
        
        path = BOOKS[book_id]["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Book not found: {path}")
        
        with open(path, 'r', errors='ignore') as f:
            return clean_gutenberg_text(f.read())
    
    def discover_from_book(self, book_id: str, max_sentences: int = 1500) -> Dict:
        """Run pattern discovery on a single book."""
        book_info = BOOKS[book_id]
        print(f"\n{'='*60}")
        print(f"Processing: {book_info['title']} by {book_info['author']}")
        print(f"{'='*60}")
        
        try:
            text = self.load_book(book_id)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            return None
        
        print(f"  Text length: {len(text):,} chars")
        
        # Run auto-discovery
        auto = AutoBootstrap(min_pattern_frequency=self.min_frequency)
        stats = auto.process_text(text, max_sentences=max_sentences)
        print(f"  Entity pairs: {stats['entity_pairs_found']}")
        
        proposed = auto.propose_patterns()
        print(f"  Patterns found: {len(proposed)}")
        
        # Store results
        result = {
            "book_id": book_id,
            "title": book_info["title"],
            "author": book_info["author"],
            "text_length": len(text),
            "entity_pairs": stats["entity_pairs_found"],
            "patterns": []
        }
        
        for p in proposed:
            pattern_info = {
                "verb": p.verb,
                "relation": p.relation_name,
                "frequency": p.frequency,
                "regex": p.regex,
                "examples": [f"{e[0]} {e[1]} {e[2]}" for e in p.examples[:3]],
            }
            result["patterns"].append(pattern_info)
            
            # Aggregate across books
            if p.verb not in self.all_patterns:
                self.all_patterns[p.verb] = {
                    "verb": p.verb,
                    "relation": p.relation_name,
                    "regex": p.regex,
                    "books": [],
                    "total_frequency": 0,
                    "examples": [],
                }
            
            self.all_patterns[p.verb]["books"].append(book_id)
            self.all_patterns[p.verb]["total_frequency"] += p.frequency
            self.all_patterns[p.verb]["examples"].extend(
                [f"{e[0]} {e[1]} {e[2]}" for e in p.examples[:2]]
            )
        
        self.book_results[book_id] = result
        
        # Show top patterns for this book
        if proposed:
            print(f"  Top patterns:")
            for p in sorted(proposed, key=lambda x: -x.frequency)[:5]:
                print(f"    {p.verb}: freq={p.frequency}")
        
        return result
    
    def discover_from_corpus(self, book_ids: List[str] = None, max_sentences: int = 1500):
        """Run discovery across multiple books."""
        if book_ids is None:
            book_ids = list(BOOKS.keys())
        
        print("\n" + "#"*60)
        print("# CORPUS-WIDE PATTERN DISCOVERY")
        print("#"*60)
        print(f"\nBooks to process: {len(book_ids)}")
        
        # Process each book
        for book_id in book_ids:
            self.discover_from_book(book_id, max_sentences=max_sentences)
        
        # Identify patterns that appear across multiple books
        self._identify_corpus_patterns()
        
        return self.corpus_patterns
    
    def _identify_corpus_patterns(self):
        """Identify patterns that appear consistently across books."""
        self.corpus_patterns = []
        
        for verb, info in self.all_patterns.items():
            num_books = len(set(info["books"]))
            
            if num_books >= self.min_books:
                pattern = {
                    "verb": verb,
                    "relation": info["relation"],
                    "regex": info["regex"],
                    "num_books": num_books,
                    "total_frequency": info["total_frequency"],
                    "avg_frequency": info["total_frequency"] / num_books,
                    "books": list(set(info["books"])),
                    "examples": info["examples"][:5],
                }
                self.corpus_patterns.append(pattern)
        
        # Sort by number of books, then by frequency
        self.corpus_patterns.sort(key=lambda x: (-x["num_books"], -x["total_frequency"]))
    
    def get_summary(self) -> Dict:
        """Get summary of corpus discovery."""
        return {
            "books_processed": len(self.book_results),
            "total_patterns_found": len(self.all_patterns),
            "corpus_patterns": len(self.corpus_patterns),
            "patterns_by_book_count": Counter(
                len(set(p["books"])) for p in self.all_patterns.values()
            ),
        }
    
    def generate_bootstrap_json(self) -> List[Dict]:
        """Generate JSON patterns for bootstrap_knowledge.json."""
        patterns = []
        
        for p in self.corpus_patterns:
            # Determine if single or two-entity pattern
            is_single = "action" in str(p["examples"]) or p["regex"].endswith("(?:\\s|$|[,.])")
            
            if is_single:
                pattern_dict = {
                    "name": p["relation"],
                    "regex": p["regex"],
                    "groups": ["entity"],
                    "fact": ["entity", p["relation"], "true"],
                    "description": f"[ENTITY] {p['verb']} -> (entity, {p['relation']}, true)",
                    "corpus_discovered": True,
                    "num_books": p["num_books"],
                    "total_frequency": p["total_frequency"],
                }
            else:
                pattern_dict = {
                    "name": p["relation"],
                    "regex": p["regex"],
                    "groups": ["entity1", "entity2"],
                    "fact": ["entity1", p["relation"], "entity2"],
                    "description": f"[ENTITY] {p['verb']} [ENTITY2] -> (entity1, {p['relation']}, entity2)",
                    "corpus_discovered": True,
                    "num_books": p["num_books"],
                    "total_frequency": p["total_frequency"],
                }
            
            patterns.append(pattern_dict)
        
        return patterns
    
    def save_results(self, filepath: str = None) -> str:
        """Save discovery results to JSON."""
        if filepath is None:
            filepath = Path(__file__).parent.parent / "truthspace_lcm" / "corpus_discovered_patterns.json"
        
        data = {
            "version": "1.0",
            "description": "Patterns discovered across literary corpus",
            "summary": self.get_summary(),
            "book_results": self.book_results,
            "corpus_patterns": self.corpus_patterns,
            "bootstrap_patterns": self.generate_bootstrap_json(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)
    
    def print_report(self):
        """Print a summary report."""
        print("\n" + "="*60)
        print("CORPUS DISCOVERY REPORT")
        print("="*60)
        
        summary = self.get_summary()
        print(f"\nBooks processed: {summary['books_processed']}")
        print(f"Total unique patterns: {summary['total_patterns_found']}")
        print(f"Corpus-wide patterns (≥{self.min_books} books): {summary['corpus_patterns']}")
        
        print(f"\nPatterns by book count:")
        for count, num in sorted(summary['patterns_by_book_count'].items(), reverse=True):
            print(f"  {count} books: {num} patterns")
        
        print(f"\n{'='*60}")
        print("TOP CORPUS-WIDE PATTERNS")
        print("="*60)
        
        for p in self.corpus_patterns[:15]:
            print(f"\n{p['verb']}:")
            print(f"  Relation: {p['relation']}")
            print(f"  Books: {p['num_books']} ({', '.join(p['books'][:3])}{'...' if len(p['books']) > 3 else ''})")
            print(f"  Total frequency: {p['total_frequency']}")
            if p['examples']:
                print(f"  Example: {p['examples'][0]}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Corpus-wide pattern discovery")
    parser.add_argument("--books", "-b", nargs="+", help="Specific books to process")
    parser.add_argument("--sentences", "-n", type=int, default=1500, help="Max sentences per book")
    parser.add_argument("--min-freq", "-f", type=int, default=3, help="Min frequency per book")
    parser.add_argument("--min-books", "-m", type=int, default=2, help="Min books for corpus pattern")
    parser.add_argument("--save", "-s", action="store_true", help="Save results to JSON")
    
    args = parser.parse_args()
    
    # Create discovery instance
    discovery = CorpusDiscovery(
        min_frequency=args.min_freq,
        min_books=args.min_books
    )
    
    # Run discovery
    book_ids = args.books if args.books else None
    discovery.discover_from_corpus(book_ids, max_sentences=args.sentences)
    
    # Print report
    discovery.print_report()
    
    # Save if requested
    if args.save:
        filepath = discovery.save_results()
        print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
