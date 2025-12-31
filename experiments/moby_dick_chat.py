#!/usr/bin/env python3
"""
Chat with Moby Dick - Quick training and interactive chat.

Trains on a portion of Moby Dick and enables interactive querying.
"""

import json
import numpy as np
import requests
import time
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.book_trainer import BookTrainer, fetch_moby_dick


def main():
    print("=" * 70)
    print("MOBY DICK CHATBOT")
    print("Training on Herman Melville's Moby Dick")
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
    
    # Initialize with seed corpus
    base = Path(__file__).parent.parent
    seed_path = base / "truthspace_lcm" / "gears" / "corpus" / "corpus_llm_live.json"
    trainer.initialize(str(seed_path) if seed_path.exists() else None)
    
    # Fetch Moby Dick
    text = fetch_moby_dick()
    if not text:
        print("Failed to fetch book!")
        return
    
    # Train on 5000 lines for faster demo
    print("\nTraining on Moby Dick (5000 lines for demo)...")
    trainer.train_on_text(
        text,
        title="Moby Dick",
        max_lines=5000,
        rebalance_every=500,
        inject_every=200,
        progress_every=1000,
    )
    
    # Print summary
    trainer.print_summary()
    
    # Interactive chat
    print(f"\n{'='*70}")
    print("CHAT WITH MOBY DICK")
    print("Ask about characters, themes, or concepts from the book.")
    print("Examples: 'Tell me about Queequeg', 'What is a whale?'")
    print("Commands: 'concepts', 'top', 'quit'")
    print(f"{'='*70}\n")
    
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not query:
            continue
        
        if query.lower() == 'quit':
            break
        
        if query.lower() == 'concepts':
            concepts = sorted(trainer.known_concepts)[:30]
            print(f"\nKnown concepts ({len(trainer.known_concepts)} total):")
            print(f"  {', '.join(concepts)}...")
            print()
            continue
        
        if query.lower() == 'top':
            print("\nTop concepts by frequency:")
            for concept, count in trainer.concept_counts.most_common(20):
                known = "✓" if concept in trainer.known_concepts else "○"
                print(f"  {known} {concept}: {count}")
            print()
            continue
        
        # Query the model
        response = trainer.query(query)
        print(f"\nBot: {response}\n")


if __name__ == "__main__":
    main()
