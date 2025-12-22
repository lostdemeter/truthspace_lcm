#!/usr/bin/env python3
"""
Concept Chat: Interactive Q&A using Holographic Concept Resolution

This script provides an interactive chat interface that:
1. Resolves questions in concept space (language-agnostic)
2. Projects answers to English using holographic projection
3. Fills the "gap" in questions with knowledge from literary works

Usage:
    python scripts/concept_chat.py
    python scripts/concept_chat.py --debug  # Show concept frames
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.concept_knowledge import ConceptQA


def main():
    parser = argparse.ArgumentParser(description='Concept Chat - Holographic Q&A')
    parser.add_argument('--debug', action='store_true', help='Show concept frames')
    parser.add_argument('--corpus', type=str, default='truthspace_lcm/concept_corpus.json',
                        help='Path to concept corpus')
    args = parser.parse_args()
    
    # Load knowledge base
    print("=" * 60)
    print("  CONCEPT CHAT - Holographic Q&A Resolution")
    print("=" * 60)
    print()
    print("Loading concept corpus...")
    
    qa = ConceptQA()
    corpus_path = Path(args.corpus)
    
    if not corpus_path.exists():
        print(f"Error: Corpus not found at {corpus_path}")
        print("Run the corpus builder first to create concept_corpus.json")
        return
    
    count = qa.load_corpus(str(corpus_path))
    print(f"Loaded {count} concept frames from literary works")
    print()
    
    # Show available entities
    entities = list(qa.knowledge.entities.keys())
    sample_entities = sorted(entities, key=lambda e: len(qa.knowledge.entities[e].get('actions', [])), reverse=True)[:10]
    print("Sample characters you can ask about:")
    for e in sample_entities:
        info = qa.knowledge.entities[e]
        actions = info.get('actions', [])
        print(f"  - {e.title()} ({len(actions)} action types)")
    print()
    
    print("Commands:")
    print("  /debug    - Toggle debug mode (show concept frames)")
    print("  /entity X - Show all info about entity X")
    print("  /stats    - Show corpus statistics")
    print("  /quit     - Exit")
    print()
    print("Ask questions like:")
    print("  'Who is Darcy?'")
    print("  'What did Holmes do?'")
    print("  'Where is Netherfield?'")
    print()
    
    debug_mode = args.debug
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith('/'):
            cmd = user_input.lower().split()[0]
            
            if cmd == '/quit':
                print("Goodbye!")
                break
            
            elif cmd == '/debug':
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
                continue
            
            elif cmd == '/stats':
                print(f"\nCorpus Statistics:")
                print(f"  Total frames: {len(qa.knowledge.frames)}")
                print(f"  Unique entities: {len(qa.knowledge.entities)}")
                print(f"  Entity relations: {len(qa.knowledge.relations)}")
                
                # Count by source
                sources = {}
                for f in qa.knowledge.frames:
                    src = f.get('source', 'unknown')
                    sources[src] = sources.get(src, 0) + 1
                print(f"\n  Frames by source:")
                for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
                    print(f"    {src}: {cnt}")
                print()
                continue
            
            elif cmd.startswith('/entity'):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: /entity <name>")
                    continue
                
                entity_name = parts[1].lower()
                info = qa.knowledge.get_entity_info(entity_name)
                
                if info:
                    print(f"\nEntity: {entity_name.title()}")
                    print(f"  Source: {info.get('source', 'unknown')}")
                    print(f"  Actions: {', '.join(info.get('actions', []))}")
                    
                    # Get frames
                    frames = qa.knowledge.query_by_entity(entity_name, k=5)
                    print(f"\n  Sample frames:")
                    for f in frames:
                        print(f"    {f.get('action', '?')}: {f.get('text', '')[:60]}...")
                else:
                    print(f"Entity '{entity_name}' not found")
                print()
                continue
            
            else:
                print(f"Unknown command: {cmd}")
                continue
        
        # Process question
        result = qa.ask_detailed(user_input)
        
        if debug_mode:
            print(f"\n[DEBUG] Axis: {result['axis']}, Entity: {result['entity']}")
        
        if result['answers']:
            best = result['answers'][0]
            
            print(f"\nBot: {best['answer']}")
            
            if debug_mode:
                print(f"\n[DEBUG] Confidence: {best['confidence']:.2f}")
                print(f"[DEBUG] Source: {best['source']}")
                print(f"[DEBUG] Frames used: {best.get('frame_count', 1)}")
                if best.get('frame'):
                    frame = best['frame']
                    print(f"[DEBUG] Frame: agent={frame.get('agent')}, action={frame.get('action')}, patient={frame.get('patient')}")
                print(f"[DEBUG] Original: {best.get('original_text', '')[:80]}...")
        else:
            print("\nBot: I don't have information about that.")
        
        print()


if __name__ == "__main__":
    main()
