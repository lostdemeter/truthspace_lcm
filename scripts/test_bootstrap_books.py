#!/usr/bin/env python3
"""
Test Bootstrap Knowledge on Multiple Gutenberg Books

This script tests the generalization of bootstrap_knowledge.json
across different books without any book-specific configuration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ingest_book import clean_gutenberg_text, split_into_sentences, BookIngester, BookChat
from truthspace_lcm.core import GeometricLCM, get_bootstrap_knowledge


# Book URLs from Project Gutenberg
BOOKS = {
    "moby_dick": {
        "url": "https://www.gutenberg.org/files/2701/2701-0.txt",
        "title": "Moby Dick",
        "author": "Herman Melville",
    },
    "pride_prejudice": {
        "url": "https://www.gutenberg.org/files/1342/1342-0.txt",
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
    },
    "frankenstein": {
        "url": "https://www.gutenberg.org/files/84/84-0.txt",
        "title": "Frankenstein",
        "author": "Mary Shelley",
    },
    "alice": {
        "url": "https://www.gutenberg.org/files/11/11-0.txt",
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
    },
}


def download_book(book_id: str) -> str:
    """Download a book from Gutenberg."""
    import urllib.request
    
    book = BOOKS[book_id]
    filepath = f"/tmp/{book_id}.txt"
    
    if not os.path.exists(filepath):
        print(f"Downloading {book['title']}...")
        urllib.request.urlretrieve(book["url"], filepath)
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def test_book(book_id: str, max_sentences: int = 500, verbose: bool = False):
    """Test bootstrap knowledge extraction on a book."""
    book = BOOKS[book_id]
    print(f"\n{'='*60}")
    print(f"Testing: {book['title']} by {book['author']}")
    print(f"{'='*60}")
    
    # Download and clean text
    raw_text = download_book(book_id)
    text = clean_gutenberg_text(raw_text)
    print(f"Text length: {len(text):,} characters")
    
    # Create ingester with bootstrap knowledge
    ingester = BookIngester(dim=256)
    
    # Ingest text
    stats = ingester.ingest_text(text, max_sentences=max_sentences, verbose=verbose)
    print(f"Sentences processed: {stats['sentences_processed']}")
    print(f"Facts extracted: {stats['relations_extracted']}")
    print(f"Entities found: {stats['entities_found']}")
    
    # Learn
    if stats['relations_extracted'] > 0:
        consistency = ingester.learn(verbose=verbose)
        print(f"Consistency: {consistency:.1%}")
    
    # Show relation types
    print("\nRelation types:")
    for rel, count in sorted(ingester.relation_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {rel}: {count}")
    
    # Show some extracted facts
    if verbose and ingester.lcm.facts:
        print("\nSample facts:")
        for fact in list(ingester.lcm.facts)[:10]:
            print(f"  {fact.subject} --{fact.relation}--> {fact.object}")
    
    return ingester.lcm, stats


def interactive_test(book_id: str):
    """Run interactive chat on a book."""
    lcm, stats = test_book(book_id, max_sentences=1000, verbose=True)
    
    if stats['relations_extracted'] == 0:
        print("\nNo facts extracted - cannot run interactive chat.")
        return
    
    chat = BookChat(lcm)
    
    print(f"\n{'='*60}")
    print(f"Interactive Chat - {BOOKS[book_id]['title']}")
    print(f"{'='*60}")
    print("\nType /help for commands, /quit to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            
            response = chat.process(user_input)
            
            if response == "QUIT":
                print("\nGoodbye!\n")
                break
            
            print(f"\nGCS: {response}\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!\n")
            break


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test bootstrap knowledge on Gutenberg books")
    parser.add_argument("book", nargs="?", choices=list(BOOKS.keys()) + ["all"],
                       default="all", help="Book to test (default: all)")
    parser.add_argument("--sentences", "-n", type=int, default=500,
                       help="Max sentences to process")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")
    parser.add_argument("--chat", "-c", action="store_true",
                       help="Start interactive chat after testing")
    
    args = parser.parse_args()
    
    if args.book == "all":
        # Test all books
        results = {}
        for book_id in BOOKS:
            try:
                _, stats = test_book(book_id, max_sentences=args.sentences, verbose=args.verbose)
                results[book_id] = stats
            except Exception as e:
                print(f"Error testing {book_id}: {e}")
                results[book_id] = {"error": str(e)}
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for book_id, stats in results.items():
            if "error" in stats:
                print(f"{BOOKS[book_id]['title']}: ERROR - {stats['error']}")
            else:
                print(f"{BOOKS[book_id]['title']}: {stats['relations_extracted']} facts from {stats['sentences_processed']} sentences")
    
    elif args.chat:
        interactive_test(args.book)
    
    else:
        test_book(args.book, max_sentences=args.sentences, verbose=args.verbose)


if __name__ == "__main__":
    main()
