#!/usr/bin/env python3
"""
Moby Dick Ingestion Test

Can we ingest an entire novel and query it meaningfully?
"""

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

from truthspace_lcm.core.semantic_space import SemanticSpace
import re
import time

def load_moby_dick(path: str) -> str:
    """Load and clean Moby Dick text."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Find start and end of actual content
    start_marker = "CHAPTER 1. Loomings."
    end_marker = "*** END OF THE PROJECT GUTENBERG"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1:
        text = text[start_idx:]
    if end_idx != -1:
        text = text[:end_idx]
    
    return text


def extract_sentences(text: str) -> list:
    """Extract sentences from text."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Clean up
    cleaned = []
    for s in sentences:
        s = s.strip()
        s = re.sub(r'\s+', ' ', s)  # Normalize whitespace
        if len(s) > 20 and len(s) < 500:  # Reasonable sentence length
            cleaned.append(s)
    
    return cleaned


def extract_paragraphs(text: str) -> list:
    """Extract paragraphs from text."""
    # Split on double newlines
    paragraphs = re.split(r'\n\s*\n', text)
    
    cleaned = []
    for p in paragraphs:
        p = p.strip()
        p = re.sub(r'\s+', ' ', p)
        if len(p) > 50 and len(p) < 1000:
            cleaned.append(p)
    
    return cleaned


def extract_chapters(text: str) -> dict:
    """Extract chapters with their content."""
    chapters = {}
    
    # Find chapter boundaries
    chapter_pattern = r'CHAPTER\s+(\d+)\.\s+([^\n]+)'
    matches = list(re.finditer(chapter_pattern, text))
    
    for i, match in enumerate(matches):
        chapter_num = match.group(1)
        chapter_title = match.group(2).strip()
        
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        content = text[start:end].strip()
        chapters[f"Chapter {chapter_num}: {chapter_title}"] = content
    
    return chapters


def main():
    print("=" * 70)
    print("MOBY DICK INGESTION TEST")
    print("=" * 70)
    
    # Load text
    print("\nLoading Moby Dick...")
    text = load_moby_dick('/tmp/moby_dick.txt')
    print(f"  Total characters: {len(text):,}")
    print(f"  Total words: {len(text.split()):,}")
    
    # Extract content
    print("\nExtracting content...")
    sentences = extract_sentences(text)
    paragraphs = extract_paragraphs(text)
    chapters = extract_chapters(text)
    
    print(f"  Sentences: {len(sentences):,}")
    print(f"  Paragraphs: {len(paragraphs):,}")
    print(f"  Chapters: {len(chapters)}")
    
    # Create semantic space with literature-appropriate seeds
    print("\nCreating SemanticSpace...")
    space = SemanticSpace(dim=64, seeds={
        # Characters
        'AHAB': ['ahab', 'captain', 'commander', 'leg', 'ivory'],
        'ISHMAEL': ['ishmael', 'narrator', 'i', 'me', 'my'],
        'QUEEQUEG': ['queequeg', 'harpooner', 'savage', 'cannibal', 'coffin'],
        'STARBUCK': ['starbuck', 'mate', 'first'],
        'STUBB': ['stubb', 'second', 'pipe'],
        
        # The whale
        'WHALE': ['whale', 'moby', 'dick', 'leviathan', 'sperm', 'white'],
        'HUNT': ['hunt', 'chase', 'pursue', 'harpoon', 'kill'],
        
        # Ship
        'SHIP': ['ship', 'pequod', 'vessel', 'boat', 'deck', 'mast'],
        'SEA': ['sea', 'ocean', 'water', 'wave', 'deep'],
        
        # Themes
        'DEATH': ['death', 'die', 'dead', 'doom', 'fate', 'destruction'],
        'OBSESSION': ['obsession', 'madness', 'revenge', 'vengeance', 'monomaniac'],
        'NATURE': ['nature', 'god', 'divine', 'creation', 'universe'],
        
        # Actions
        'SAIL': ['sail', 'sailing', 'voyage', 'journey', 'travel'],
        'SPEAK': ['said', 'say', 'spoke', 'cried', 'exclaimed'],
    })
    
    # Ingest paragraphs (more manageable than sentences)
    print(f"\nIngesting {len(paragraphs)} paragraphs...")
    start_time = time.time()
    
    for i, para in enumerate(paragraphs):
        # Store the paragraph with itself as the value (for retrieval)
        space[para] = para
        
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  {i + 1:,} paragraphs ({rate:.1f}/sec)")
    
    elapsed = time.time() - start_time
    print(f"  Done! {len(paragraphs):,} paragraphs in {elapsed:.1f}s")
    print(f"  Vocabulary size: {len(space._word_positions):,} words")
    
    # Test queries
    print("\n" + "=" * 70)
    print("QUERY TESTS")
    print("=" * 70)
    
    queries = [
        "Who is Captain Ahab?",
        "What is the white whale?",
        "Tell me about Queequeg",
        "What happened to the Pequod?",
        "Why does Ahab hunt the whale?",
        "Describe the sea",
        "What is Ishmael's story?",
        "How does the book begin?",
        "What is the fate of the crew?",
        "Tell me about harpooning whales",
    ]
    
    for query in queries:
        result, score = space.get_with_score(query)
        print(f"\nQ: {query}")
        print(f"   Score: {score:.3f}")
        if result:
            # Show first 150 chars of the matching paragraph
            preview = result[:150] + "..." if len(result) > 150 else result
            print(f"   A: {preview}")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("=" * 70)
    
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not query or query.lower() == 'quit':
            break
        
        # Get top 3 matches
        matches = space.top_k(query, k=3)
        
        print(f"\nTop matches:")
        for i, (key, value, score) in enumerate(matches):
            preview = key[:200] + "..." if len(key) > 200 else key
            print(f"\n  [{i+1}] Score: {score:.3f}")
            print(f"      {preview}")


if __name__ == "__main__":
    main()
