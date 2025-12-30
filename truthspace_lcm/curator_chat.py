#!/usr/bin/env python3
"""
Curator Chatbot: Interactive Corpus Improvement

An interactive chatbot that helps curate and improve corpus quality.
Uses the CuratorLCM to score sentences and suggest improvements.

Commands:
    /score <sentence>  - Score a sentence for frame quality
    /frame <sentence>  - Extract and score the frame
    /rewrite <sentence> - Suggest a rewrite
    /batch <file>      - Score all sentences in a file
    /stats             - Show curator statistics
    /help              - Show help

Usage:
    python -m truthspace_lcm.curator_chat
    python -m truthspace_lcm.curator_chat --corpus path/to/corpus.json

Author: Lesley Gushurst
License: GPLv3
"""

import argparse
import sys
from pathlib import Path

from .core.curator import CuratorLCM
from .core.geometric import GeometricQA, GeometricKnowledge


def main():
    parser = argparse.ArgumentParser(description='Curator Chatbot')
    parser.add_argument('--corpus', type=str, help='Corpus to load for knowledge')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Curator LCM - Interactive Corpus Improvement")
    print("  Self-improving curation using geometric features")
    print("=" * 60)
    print()
    
    # Load knowledge if corpus provided
    knowledge = None
    if args.corpus:
        corpus_path = Path(args.corpus)
        if corpus_path.exists():
            print(f"Loading corpus from {corpus_path}...")
            qa = GeometricQA()
            qa.load_corpus(str(corpus_path))
            knowledge = qa.knowledge
            print(f"Loaded {len(knowledge.frames)} frames, {len(knowledge.concepts)} concepts")
        else:
            print(f"Corpus not found: {corpus_path}")
    else:
        # Try default corpus
        default_path = Path(__file__).parent / 'sample_corpus_geometric.json'
        if default_path.exists():
            print(f"Loading default corpus...")
            qa = GeometricQA()
            qa.load_corpus(str(default_path))
            knowledge = qa.knowledge
            print(f"Loaded {len(knowledge.frames)} frames")
    
    # Create curator
    curator = CuratorLCM(knowledge)
    
    if knowledge:
        print(f"Learned {len(curator.learned_initiators)} initiators, "
              f"{len(curator.learned_mediators)} mediators, "
              f"{len(curator.learned_receivers)} receivers")
    
    print()
    print("Commands:")
    print("  /score <sentence>  - Score a sentence")
    print("  /frame <sentence>  - Extract and score frame")
    print("  /rewrite <sentence> - Suggest rewrite")
    print("  /explain <sentence> - Detailed explanation")
    print("  /batch <file>      - Score sentences from file")
    print("  /help              - Show this help")
    print("  /quit              - Exit")
    print()
    print("Or just type a sentence to score it.")
    print()
    
    # Statistics
    total_scored = 0
    total_good = 0
    total_acceptable = 0
    total_poor = 0
    
    while True:
        try:
            user_input = input("Curator> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith('/'):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if cmd == '/quit' or cmd == '/exit':
                print("Goodbye!")
                break
            
            elif cmd == '/help':
                print("\nCommands:")
                print("  /score <sentence>  - Score a sentence for frame quality")
                print("  /frame <sentence>  - Extract frame and score it")
                print("  /rewrite <sentence> - Suggest a better version")
                print("  /explain <sentence> - Detailed scoring explanation")
                print("  /batch <file>      - Score all sentences in a file")
                print("  /stats             - Show scoring statistics")
                print("  /quit              - Exit")
                print()
                continue
            
            elif cmd == '/stats':
                print(f"\nScoring Statistics:")
                print(f"  Total scored: {total_scored}")
                print(f"  Good (≥0.7): {total_good} ({100*total_good/max(1,total_scored):.1f}%)")
                print(f"  Acceptable (0.5-0.7): {total_acceptable} ({100*total_acceptable/max(1,total_scored):.1f}%)")
                print(f"  Poor (<0.5): {total_poor} ({100*total_poor/max(1,total_scored):.1f}%)")
                print()
                continue
            
            elif cmd == '/score':
                if not arg:
                    print("Usage: /score <sentence>")
                    continue
                sentence = arg
            
            elif cmd == '/explain':
                if not arg:
                    print("Usage: /explain <sentence>")
                    continue
                print()
                print(curator.explain_score(arg))
                print()
                continue
            
            elif cmd == '/rewrite':
                if not arg:
                    print("Usage: /rewrite <sentence>")
                    continue
                rewrite = curator.suggest_rewrite(arg)
                if rewrite:
                    print(f"\nOriginal: {arg}")
                    print(f"Rewrite:  {rewrite}")
                    
                    # Score both
                    orig_score = curator.score_sentence(arg)
                    new_score = curator.score_sentence(rewrite)
                    print(f"\nOriginal score: {orig_score.overall:.2f}")
                    print(f"Rewrite score:  {new_score.overall:.2f}")
                    
                    if new_score.overall > orig_score.overall:
                        print("✓ Improvement!")
                    else:
                        print("~ No improvement (original may be fine)")
                else:
                    print("No rewrite suggested (sentence may be good enough)")
                print()
                continue
            
            elif cmd == '/frame':
                if not arg:
                    print("Usage: /frame <sentence>")
                    continue
                
                # Extract frame using geometric knowledge
                if knowledge:
                    # Simple frame extraction for demo
                    words = arg.lower().split()
                    n = len(words)
                    
                    # Position-based extraction
                    initiator = words[0] if n > 0 else ""
                    mediator = words[n//2] if n > 1 else ""
                    receiver = words[-1] if n > 2 else ""
                    
                    frame_score = curator.score_frame(initiator, mediator, receiver, arg)
                    
                    print(f"\nFrame: {initiator} | {mediator} | {receiver}")
                    print(f"Score: {frame_score.overall:.2f}")
                    if frame_score.issues:
                        print(f"Issues: {', '.join(frame_score.issues)}")
                else:
                    print("No knowledge loaded - can't extract frame")
                print()
                continue
            
            elif cmd == '/batch':
                if not arg:
                    print("Usage: /batch <file>")
                    continue
                
                batch_path = Path(arg)
                if not batch_path.exists():
                    print(f"File not found: {arg}")
                    continue
                
                sentences = batch_path.read_text().strip().split('\n')
                good = acceptable = poor = 0
                
                print(f"\nScoring {len(sentences)} sentences...")
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    score = curator.score_sentence(s)
                    if score.overall >= 0.7:
                        good += 1
                    elif score.overall >= 0.5:
                        acceptable += 1
                    else:
                        poor += 1
                
                print(f"Results:")
                print(f"  Good (≥0.7): {good}")
                print(f"  Acceptable (0.5-0.7): {acceptable}")
                print(f"  Poor (<0.5): {poor}")
                print()
                continue
            
            else:
                print(f"Unknown command: {cmd}")
                print("Type /help for available commands")
                continue
        
        else:
            # Score the sentence directly
            sentence = user_input
        
        # Score the sentence
        score = curator.score_sentence(sentence)
        total_scored += 1
        
        if score.overall >= 0.7:
            verdict = "✓ Good"
            total_good += 1
        elif score.overall >= 0.5:
            verdict = "~ Acceptable"
            total_acceptable += 1
        else:
            verdict = "✗ Poor"
            total_poor += 1
        
        print(f"\n{verdict} [{score.overall:.2f}]")
        
        if score.issues:
            print(f"Issues: {', '.join(score.issues)}")
        
        if score.suggestions:
            print(f"Suggestions: {', '.join(score.suggestions)}")
        
        # Suggest rewrite for poor sentences
        if score.overall < 0.5:
            rewrite = curator.suggest_rewrite(sentence)
            if rewrite:
                print(f"Suggested rewrite: {rewrite}")
        
        print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
