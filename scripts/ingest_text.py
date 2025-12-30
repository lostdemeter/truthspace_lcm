#!/usr/bin/env python3
"""
Text Ingestion Pipeline for Corpus Scaling

Ingest text from various sources while maintaining geometric principles:
- Position-based frame extraction
- Morphology learning from parallel structures
- Semantic quaternion population

Supports:
- Plain text files
- Project Gutenberg books
- Wikipedia articles (via API)
- JSON corpora

Usage:
    python scripts/ingest_text.py --file book.txt --source "Pride and Prejudice"
    python scripts/ingest_text.py --gutenberg 1342  # Pride and Prejudice
    python scripts/ingest_text.py --wikipedia "Sherlock Holmes"
    python scripts/ingest_text.py --directory ./books/

Author: Lesley Gushurst
License: GPLv3
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.geometric import GeometricKnowledge, GeometricQA


def clean_text(text: str) -> str:
    """Clean text for processing."""
    # Remove Project Gutenberg headers/footers
    if "*** START OF" in text:
        start = text.find("*** START OF")
        end_marker = text.find("***", start + 10)
        if end_marker > start:
            text = text[end_marker + 3:]
    
    if "*** END OF" in text:
        end = text.find("*** END OF")
        text = text[:end]
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove chapter markers
    text = re.sub(r'CHAPTER [IVXLCDM]+\.?', '', text)
    text = re.sub(r'Chapter \d+', '', text)
    
    return text.strip()


def is_good_sentence(s: str) -> bool:
    """Check if a sentence is suitable for frame extraction."""
    words = s.split()
    
    # Length check: 6-25 words is ideal for frame extraction
    if not (6 <= len(words) <= 25):
        return False
    
    # Must start with capital letter
    if not s[0].isupper():
        return False
    
    # Skip sentences that are mostly punctuation or numbers
    alpha_words = [w for w in words if w.isalpha()]
    if len(alpha_words) < len(words) * 0.8:
        return False
    
    # Must have a subject-verb structure (simple heuristic)
    # First word should be a noun/pronoun (capitalized or common pronoun)
    first_word = words[0].lower()
    good_starters = {'i', 'he', 'she', 'it', 'we', 'they', 'the', 'a', 'an', 'my', 'his', 'her', 'our', 'their'}
    if first_word not in good_starters and not words[0][0].isupper():
        return False
    
    # Must have a verb (look for common patterns)
    verb_indicators = ('ed', 'ing', 'es')
    common_verbs = {'is', 'are', 'was', 'were', 'had', 'has', 'have', 'did', 'do', 'does',
                    'said', 'told', 'asked', 'went', 'came', 'saw', 'knew', 'thought',
                    'looked', 'found', 'made', 'took', 'gave', 'got', 'put', 'let'}
    
    has_verb = any(
        w.lower() in common_verbs or 
        (w.endswith(verb_indicators) and len(w) > 3)
        for w in words[1:]  # Skip first word
    )
    if not has_verb:
        return False
    
    # Skip dialogue markers and fragments
    if s.startswith(('"', "'", '--', '—')):
        return False
    
    # Skip sentences with too many proper nouns (likely lists or titles)
    proper_nouns = sum(1 for w in words if w[0].isupper() and w.lower() not in {'i'})
    if proper_nouns > len(words) * 0.5:
        return False
    
    return True


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Handle abbreviations
    text = re.sub(r'Mr\.', 'Mr', text)
    text = re.sub(r'Mrs\.', 'Mrs', text)
    text = re.sub(r'Dr\.', 'Dr', text)
    text = re.sub(r'St\.', 'St', text)
    text = re.sub(r'No\.', 'No', text)
    
    # Remove dialogue markers
    text = re.sub(r'["\']', '', text)
    
    # Split on sentence boundaries
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Filter for quality
    result = [s.strip() for s in sentences if is_good_sentence(s.strip())]
    
    return result


def ingest_text(knowledge: GeometricKnowledge, text: str, source: str, 
                max_sentences: int = 10000, verbose: bool = False,
                use_curator: bool = True, min_score: float = 0.5) -> int:
    """
    Ingest text into geometric knowledge.
    
    Args:
        knowledge: GeometricKnowledge to add to
        text: Raw text to process
        source: Source name
        max_sentences: Maximum sentences to process
        verbose: Print progress
        use_curator: Use CuratorLCM to filter sentences
        min_score: Minimum curator score to accept (0-1)
    
    Returns number of sentences processed.
    """
    text = clean_text(text)
    sentences = split_sentences(text)
    
    if verbose:
        print(f"Found {len(sentences)} candidate sentences in {source}")
    
    # Use curator to filter if enabled
    if use_curator:
        from truthspace_lcm.core.curator import CuratorLCM
        curator = CuratorLCM(knowledge)  # Use existing knowledge to improve scoring
        
        accepted = []
        rejected = 0
        for s in sentences:
            score = curator.score_sentence(s)
            if score.overall >= min_score:
                accepted.append(s)
            else:
                rejected += 1
        
        if verbose:
            print(f"  Curator accepted {len(accepted)}, rejected {rejected} (min_score={min_score})")
        
        sentences = accepted
    
    count = 0
    for sentence in sentences[:max_sentences]:
        knowledge.learn(sentence, source)
        count += 1
        
        if verbose and count % 1000 == 0:
            print(f"  Processed {count} sentences...")
    
    return count


def ingest_file(knowledge: GeometricKnowledge, path: Path, source: str = None,
                max_sentences: int = 10000, verbose: bool = False) -> int:
    """Ingest a text file."""
    if source is None:
        source = path.stem
    
    if verbose:
        print(f"Reading {path}...")
    
    text = path.read_text(encoding='utf-8', errors='ignore')
    return ingest_text(knowledge, text, source, max_sentences, verbose)


def ingest_gutenberg(knowledge: GeometricKnowledge, book_id: int,
                     max_sentences: int = 10000, verbose: bool = False) -> int:
    """
    Ingest a book from Project Gutenberg.
    
    Common book IDs:
    - 1342: Pride and Prejudice
    - 1661: Sherlock Holmes
    - 84: Frankenstein
    - 1232: The Prince
    - 2701: Moby Dick
    """
    import urllib.request
    
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    alt_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    
    if verbose:
        print(f"Downloading Gutenberg book {book_id}...")
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            text = response.read().decode('utf-8', errors='ignore')
    except:
        try:
            with urllib.request.urlopen(alt_url, timeout=30) as response:
                text = response.read().decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Error downloading book {book_id}: {e}")
            return 0
    
    # Extract title from text
    title_match = re.search(r'Title:\s*(.+)', text)
    source = title_match.group(1).strip() if title_match else f"Gutenberg-{book_id}"
    
    if verbose:
        print(f"Book: {source}")
    
    return ingest_text(knowledge, text, source, max_sentences, verbose)


def ingest_grokipedia(knowledge: GeometricKnowledge, topic: str,
                      max_sentences: int = 1000, verbose: bool = False,
                      use_curator: bool = True, min_score: float = 0.5) -> int:
    """
    Ingest a Grokipedia article.
    
    Uses grokipedia-api.com (100 req/min, no API key needed).
    """
    import urllib.request
    
    # Convert spaces to underscores for URL
    topic_slug = topic.replace(' ', '_')
    url = f"https://grokipedia-api.com/page/{topic_slug}"
    
    if verbose:
        print(f"Fetching Grokipedia article: {topic}...")
    
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'TruthSpaceLCM/1.0 (corpus builder)'}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Error fetching Grokipedia article: {e}")
        return 0
    
    # Extract text (API returns 'content_text')
    text = data.get("content_text", "") or data.get("text", "") or ""
    
    if not text:
        print(f"No text found for Grokipedia article: {topic}")
        return 0
    
    return ingest_text(knowledge, text, f"Grokipedia: {topic}", max_sentences, verbose, use_curator, min_score)


def ingest_directory(knowledge: GeometricKnowledge, directory: Path,
                     max_sentences_per_file: int = 5000, verbose: bool = False) -> int:
    """Ingest all text files in a directory."""
    total = 0
    
    for path in directory.glob("*.txt"):
        count = ingest_file(knowledge, path, max_sentences=max_sentences_per_file, verbose=verbose)
        total += count
    
    return total


def save_corpus(knowledge: GeometricKnowledge, output_path: Path, verbose: bool = False):
    """Save the knowledge as a JSON corpus."""
    corpus = {
        "frames": [],
        "metadata": {
            "total_sentences": knowledge.total_sentences,
            "total_concepts": len(knowledge.concepts),
            "morphology_clusters": len(knowledge.morphology.equivalence_classes),
        }
    }
    
    for frame in knowledge.frames:
        corpus["frames"].append({
            "initiator": frame.initiator,
            "mediator": frame.mediator,
            "receiver": frame.receiver,
            "source": frame.source,
            "text": frame.text,
        })
    
    if verbose:
        print(f"Saving {len(corpus['frames'])} frames to {output_path}...")
    
    with open(output_path, 'w') as f:
        json.dump(corpus, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Ingest text into geometric corpus')
    
    # Input sources
    parser.add_argument('--file', type=str, help='Text file to ingest')
    parser.add_argument('--gutenberg', type=int, help='Project Gutenberg book ID')
    parser.add_argument('--grokipedia', type=str, help='Grokipedia article topic (e.g., "Sherlock_Holmes")')
    parser.add_argument('--directory', type=str, help='Directory of text files')
    parser.add_argument('--source', type=str, help='Source name for the text')
    
    # Options
    parser.add_argument('--max-sentences', type=int, default=10000,
                        help='Maximum sentences to process per source')
    parser.add_argument('--output', type=str, default='corpus_expanded.json',
                        help='Output corpus file')
    parser.add_argument('--append', type=str, help='Existing corpus to append to')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize knowledge
    knowledge = GeometricKnowledge()
    
    # Load existing corpus if appending
    if args.append:
        append_path = Path(args.append)
        if append_path.exists():
            qa = GeometricQA()
            qa.load_corpus(str(append_path))
            knowledge = qa.knowledge
            if args.verbose:
                print(f"Loaded existing corpus: {len(knowledge.frames)} frames")
    
    # Process inputs
    total_sentences = 0
    
    if args.file:
        count = ingest_file(knowledge, Path(args.file), args.source, 
                           args.max_sentences, args.verbose)
        total_sentences += count
    
    if args.gutenberg:
        count = ingest_gutenberg(knowledge, args.gutenberg, 
                                args.max_sentences, args.verbose)
        total_sentences += count
    
    if args.grokipedia:
        count = ingest_grokipedia(knowledge, args.grokipedia,
                                  args.max_sentences, args.verbose)
        total_sentences += count
    
    if args.directory:
        count = ingest_directory(knowledge, Path(args.directory),
                                args.max_sentences, args.verbose)
        total_sentences += count
    
    if total_sentences == 0:
        print("No input specified. Use --file, --gutenberg, --wikipedia, or --directory")
        parser.print_help()
        return 1
    
    # Save output
    output_path = Path(args.output)
    save_corpus(knowledge, output_path, args.verbose)
    
    # Summary
    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Sentences processed: {total_sentences}")
    print(f"Total frames: {len(knowledge.frames)}")
    print(f"Total concepts: {len(knowledge.concepts)}")
    print(f"Content words: {len([c for c in knowledge.concepts.values() if c.is_content_word])}")
    print(f"Morphology clusters: {len(knowledge.morphology.equivalence_classes)}")
    print(f"Output: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
