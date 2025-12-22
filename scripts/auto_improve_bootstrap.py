#!/usr/bin/env python3
"""
Auto-Improving Bootstrap System

This script implements an iterative improvement loop that:
1. Processes text to discover entity-verb-entity patterns
2. Proposes new patterns from frequent co-occurrences
3. Validates patterns (optionally with human feedback)
4. Merges validated patterns into bootstrap_knowledge.json
5. Re-tests extraction with expanded patterns
6. Repeats until convergence

Usage:
    # Auto mode (no human validation)
    python scripts/auto_improve_bootstrap.py --auto
    
    # Interactive mode (human validates each pattern)
    python scripts/auto_improve_bootstrap.py --interactive
    
    # Single iteration
    python scripts/auto_improve_bootstrap.py --iterations 1
    
    # Use specific book
    python scripts/auto_improve_bootstrap.py --book /tmp/pride_prejudice.txt
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core.auto_bootstrap import AutoBootstrap
from truthspace_lcm.core.bootstrap import BootstrapKnowledge
from scripts.ingest_book import clean_gutenberg_text, split_into_sentences


def download_book(url: str, filepath: str) -> str:
    """Download a book from Project Gutenberg."""
    import urllib.request
    
    if not os.path.exists(filepath):
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, filepath)
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def load_texts(book_paths: list = None) -> str:
    """Load and combine texts from multiple sources."""
    if book_paths is None:
        # Default: use available Gutenberg books
        book_paths = [
            '/tmp/pride_prejudice.txt',
            '/tmp/alice.txt', 
            '/tmp/frankenstein.txt',
            '/tmp/moby_dick.txt',
        ]
    
    texts = []
    for path in book_paths:
        if os.path.exists(path):
            with open(path, 'r', errors='ignore') as f:
                text = clean_gutenberg_text(f.read())
                texts.append(text)
                print(f"  Loaded {path}: {len(text):,} chars")
    
    return ' '.join(texts)


def count_extractions(text: str, max_sentences: int = 500) -> dict:
    """Count how many facts we can extract with current bootstrap knowledge."""
    bootstrap = BootstrapKnowledge()
    sentences = split_into_sentences(text)[:max_sentences]
    
    total_facts = 0
    relation_counts = {}
    
    for sentence in sentences:
        facts = bootstrap.extract_facts(sentence, use_coreference=False)
        total_facts += len(facts)
        for fact in facts:
            relation_counts[fact.relation] = relation_counts.get(fact.relation, 0) + 1
    
    return {
        'sentences': len(sentences),
        'facts': total_facts,
        'facts_per_sentence': total_facts / len(sentences) if sentences else 0,
        'relations': relation_counts,
    }


def run_improvement_iteration(text: str, max_sentences: int = 2000, 
                               min_frequency: int = 3, interactive: bool = False) -> dict:
    """Run one iteration of the improvement loop."""
    
    # Phase 1: Discover patterns
    print("\n" + "="*60)
    print("PHASE 1: Pattern Discovery")
    print("="*60)
    
    auto = AutoBootstrap(min_pattern_frequency=min_frequency)
    stats = auto.process_text(text, max_sentences=max_sentences)
    print(f"  Entity pairs found: {stats['entity_pairs_found']}")
    print(f"  Unique verbs: {stats['unique_verbs']}")
    
    # Phase 2: Propose patterns
    print("\n" + "="*60)
    print("PHASE 2: Pattern Proposal")
    print("="*60)
    
    proposed = auto.propose_patterns()
    print(f"  Patterns proposed: {len(proposed)}")
    
    if not proposed:
        print("  No new patterns found.")
        return {'new_patterns': 0, 'merged': 0}
    
    for p in proposed:
        print(f"    - {p.verb}: freq={p.frequency}")
    
    # Phase 3: Validation
    print("\n" + "="*60)
    print("PHASE 3: Pattern Validation")
    print("="*60)
    
    if interactive:
        validated = auto.interactive_validation()
    else:
        # Auto-validate all proposed patterns
        auto.validated_patterns = proposed
        for p in proposed:
            p.validated = True
        validated = proposed
        print(f"  Auto-validated {len(validated)} patterns")
    
    if not validated:
        print("  No patterns validated.")
        return {'new_patterns': len(proposed), 'merged': 0}
    
    # Phase 4: Merge with bootstrap
    print("\n" + "="*60)
    print("PHASE 4: Merge with Bootstrap Knowledge")
    print("="*60)
    
    result = auto.merge_with_bootstrap()
    print(f"  Existing patterns: {result['existing_patterns']}")
    print(f"  New patterns added: {result['new_patterns_added']}")
    print(f"  Total patterns: {result['total_patterns']}")
    
    return {
        'new_patterns': len(proposed),
        'validated': len(validated),
        'merged': result['new_patterns_added'],
    }


def run_improvement_loop(text: str, max_iterations: int = 5, 
                         interactive: bool = False, target_improvement: float = 0.1):
    """
    Run the full improvement loop until convergence.
    
    Stops when:
    - No new patterns are discovered
    - Extraction rate stops improving
    - Max iterations reached
    """
    print("\n" + "="*60)
    print("AUTO-IMPROVING BOOTSTRAP SYSTEM")
    print("="*60)
    
    # Baseline measurement
    print("\nBaseline extraction rate:")
    baseline = count_extractions(text)
    print(f"  {baseline['facts']} facts from {baseline['sentences']} sentences")
    print(f"  Rate: {baseline['facts_per_sentence']:.3f} facts/sentence")
    
    best_rate = baseline['facts_per_sentence']
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'#'*60}")
        print(f"# ITERATION {iteration}")
        print(f"{'#'*60}")
        
        # Run improvement
        result = run_improvement_iteration(
            text, 
            max_sentences=2000,
            min_frequency=max(2, 5 - iteration),  # Lower threshold each iteration
            interactive=interactive
        )
        
        if result['merged'] == 0:
            print("\nNo new patterns merged. Stopping.")
            break
        
        # Measure improvement
        print("\n" + "="*60)
        print("MEASURING IMPROVEMENT")
        print("="*60)
        
        # Reload bootstrap to get new patterns
        current = count_extractions(text)
        improvement = current['facts_per_sentence'] - best_rate
        
        print(f"  Previous rate: {best_rate:.3f} facts/sentence")
        print(f"  Current rate: {current['facts_per_sentence']:.3f} facts/sentence")
        print(f"  Improvement: {improvement:+.3f}")
        
        if current['facts_per_sentence'] > best_rate:
            best_rate = current['facts_per_sentence']
        
        if improvement < target_improvement and iteration > 1:
            print("\nImprovement below threshold. Stopping.")
            break
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    final = count_extractions(text)
    total_improvement = final['facts_per_sentence'] - baseline['facts_per_sentence']
    
    print(f"  Iterations: {iteration}")
    print(f"  Baseline: {baseline['facts_per_sentence']:.3f} facts/sentence")
    print(f"  Final: {final['facts_per_sentence']:.3f} facts/sentence")
    print(f"  Total improvement: {total_improvement:+.3f} ({total_improvement/baseline['facts_per_sentence']*100:+.1f}%)")
    print(f"\nRelation types found:")
    for rel, count in sorted(final['relations'].items(), key=lambda x: -x[1])[:10]:
        print(f"    {rel}: {count}")
    
    return {
        'iterations': iteration,
        'baseline_rate': baseline['facts_per_sentence'],
        'final_rate': final['facts_per_sentence'],
        'improvement': total_improvement,
    }


def main():
    parser = argparse.ArgumentParser(description="Auto-improve bootstrap knowledge")
    parser.add_argument("--book", "-b", type=str, help="Path to book file")
    parser.add_argument("--iterations", "-n", type=int, default=5, help="Max iterations")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive validation")
    parser.add_argument("--auto", "-a", action="store_true", help="Auto mode (no interaction)")
    parser.add_argument("--min-freq", "-f", type=int, default=3, help="Min pattern frequency")
    
    args = parser.parse_args()
    
    # Load text
    print("Loading texts...")
    if args.book:
        with open(args.book, 'r', errors='ignore') as f:
            text = clean_gutenberg_text(f.read())
        print(f"  Loaded {args.book}: {len(text):,} chars")
    else:
        text = load_texts()
    
    if not text:
        print("No text loaded. Please provide a book file or download Gutenberg books first.")
        return
    
    # Run improvement loop
    result = run_improvement_loop(
        text,
        max_iterations=args.iterations,
        interactive=args.interactive and not args.auto,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
