#!/usr/bin/env python3
"""
TruthSpace LCM Chat Interface

Holographic Concept Q&A using language-agnostic concept frames.

Architecture:
    Question (any language)
            ↓
    Axis Detection (WHO/WHAT/WHERE/etc)
            ↓
    Concept Space Query (language-agnostic)
            ↓
    Knowledge Aggregation
            ↓
    Holographic Projection (fill the gap)
            ↓
    English Answer

Holographic Principle:
    Question = Content - Gap    (has missing information)
    Answer   = Content + Fill   (provides missing information)

2D φ-Dial Control:
    "We control the horizontal. We control the vertical."
    
    Horizontal (x): Style
        -1 = formal, specific, rare
        +1 = casual, universal, common
        
    Vertical (y): Perspective
        -1 = subjective, experiential
         0 = objective, factual
        +1 = meta, analytical

Usage:
    python -m truthspace_lcm.chat
    python -m truthspace_lcm.chat --debug
    python -m truthspace_lcm.chat --style -1 --perspective 1  # Formal + Meta
    python -m truthspace_lcm.chat --corpus path/to/corpus.json
"""

import argparse
import sys
from pathlib import Path

from .core import ConceptQA


def main():
    parser = argparse.ArgumentParser(
        description='TruthSpace LCM - Holographic Concept Q&A'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Show concept frames and debug info'
    )
    parser.add_argument(
        '--corpus', type=str, default=None,
        help='Path to concept corpus JSON file'
    )
    parser.add_argument(
        '--style', '-x', type=float, default=0.0,
        help='Style dial: -1 (formal) to +1 (casual)'
    )
    parser.add_argument(
        '--perspective', '-y', type=float, default=0.0,
        help='Perspective dial: -1 (subjective) to +1 (meta)'
    )
    parser.add_argument(
        '--depth', '-z', type=float, default=0.0,
        help='Depth dial: -1 (terse) to +1 (elaborate)'
    )
    args = parser.parse_args()
    
    # Find corpus
    if args.corpus:
        corpus_path = Path(args.corpus)
    else:
        # Default: look in package directory
        corpus_path = Path(__file__).parent / 'concept_corpus.json'
    
    # Initialize Q&A system
    print("=" * 60)
    print("  TruthSpace LCM - Holographic Concept Q&A")
    print("=" * 60)
    print()
    
    if not corpus_path.exists():
        print(f"Error: Corpus not found at {corpus_path}")
        print()
        print("To build a corpus, run:")
        print("  python scripts/build_concept_corpus.py")
        return 1
    
    print(f"Loading corpus from {corpus_path}...")
    qa = ConceptQA(style_x=args.style, perspective_y=args.perspective, depth_z=args.depth)
    count = qa.load_corpus(str(corpus_path))
    print(f"Loaded {count} concept frames")
    
    # Show dial settings
    style_label = 'formal' if args.style < -0.3 else ('casual' if args.style > 0.3 else 'neutral')
    perspective_label = 'subjective' if args.perspective < -0.3 else ('meta' if args.perspective > 0.3 else 'objective')
    depth_label = 'terse' if args.depth < -0.3 else ('elaborate' if args.depth > 0.3 else 'standard')
    print(f"φ-Dial: style={style_label} (x={args.style:+.1f}), perspective={perspective_label} (y={args.perspective:+.1f}), depth={depth_label} (z={args.depth:+.1f})")
    print()
    
    # Show sample entities
    entities = list(qa.knowledge.entities.keys())
    top_entities = sorted(
        entities,
        key=lambda e: len(qa.knowledge.entities[e].get('actions', [])),
        reverse=True
    )[:8]
    
    print("Sample characters:")
    for e in top_entities:
        info = qa.knowledge.entities[e]
        actions = info.get('actions', [])
        source = info.get('source', 'unknown')
        print(f"  - {e.title()} ({source})")
    print()
    
    print("Commands:")
    print("  /debug      - Toggle debug mode")
    print("  /entity X   - Show info about entity X")
    print("  /stats      - Show corpus statistics")
    print("  /style X    - Set style dial (-1=formal, +1=casual)")
    print("  /perspective Y - Set perspective dial (-1=subjective, +1=meta)")
    print("  /depth Z    - Set depth dial (-1=terse, +1=elaborate)")
    print("  /dial       - Show current dial settings")
    print("  /help       - Show this help")
    print("  /quit       - Exit")
    print()
    print("Ask questions like:")
    print('  "Who is Darcy?"')
    print('  "What did Holmes do?"')
    print('  "Where is Netherfield?"')
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
            
            if cmd == '/quit' or cmd == '/exit':
                print("Goodbye!")
                break
            
            elif cmd == '/debug':
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
                continue
            
            elif cmd == '/help':
                print("\nCommands:")
                print("  /debug      - Toggle debug mode")
                print("  /entity X   - Show info about entity X")
                print("  /stats      - Show corpus statistics")
                print("  /style X    - Set style dial (-1=formal, +1=casual)")
                print("  /perspective Y - Set perspective dial (-1=subjective, +1=meta)")
                print("  /depth Z    - Set depth dial (-1=terse, +1=elaborate)")
                print("  /dial       - Show current dial settings")
                print("  /quit       - Exit")
                print()
                continue
            
            elif cmd == '/dial':
                style_label = 'formal' if qa.style_x < -0.3 else ('casual' if qa.style_x > 0.3 else 'neutral')
                perspective_label = 'subjective' if qa.perspective_y < -0.3 else ('meta' if qa.perspective_y > 0.3 else 'objective')
                depth_label = 'terse' if qa.depth_z < -0.3 else ('elaborate' if qa.depth_z > 0.3 else 'standard')
                print(f"\nφ-Dial Settings:")
                print(f"  Style (x):       {qa.style_x:+.1f} ({style_label})")
                print(f"  Perspective (y): {qa.perspective_y:+.1f} ({perspective_label})")
                print(f"  Depth (z):       {qa.depth_z:+.1f} ({depth_label})")
                print(f"  Octant: {qa.projector.answer_generator.dial.get_octant_label()}")
                print()
                continue
            
            elif cmd.startswith('/style'):
                parts = user_input.split()
                if len(parts) < 2:
                    print("Usage: /style <value>  (e.g., /style -1 for formal, /style 1 for casual)")
                    continue
                try:
                    x = float(parts[1])
                    qa.set_style(x)
                    style_label = 'formal' if x < -0.3 else ('casual' if x > 0.3 else 'neutral')
                    print(f"Style set to {x:+.1f} ({style_label})")
                except ValueError:
                    print("Invalid value. Use a number between -1 and 1.")
                continue
            
            elif cmd.startswith('/perspective'):
                parts = user_input.split()
                if len(parts) < 2:
                    print("Usage: /perspective <value>  (e.g., /perspective -1 for subjective, /perspective 1 for meta)")
                    continue
                try:
                    y = float(parts[1])
                    qa.set_perspective(y)
                    perspective_label = 'subjective' if y < -0.3 else ('meta' if y > 0.3 else 'objective')
                    print(f"Perspective set to {y:+.1f} ({perspective_label})")
                except ValueError:
                    print("Invalid value. Use a number between -1 and 1.")
                continue
            
            elif cmd.startswith('/depth'):
                parts = user_input.split()
                if len(parts) < 2:
                    print("Usage: /depth <value>  (e.g., /depth -1 for terse, /depth 1 for elaborate)")
                    continue
                try:
                    z = float(parts[1])
                    qa.set_depth(z)
                    depth_label = 'terse' if z < -0.3 else ('elaborate' if z > 0.3 else 'standard')
                    print(f"Depth set to {z:+.1f} ({depth_label})")
                except ValueError:
                    print("Invalid value. Use a number between -1 and 1.")
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
                    
                    # Get sample frames
                    frames = qa.knowledge.query_by_entity(entity_name, k=5)
                    if frames:
                        print(f"\n  Sample occurrences:")
                        for f in frames:
                            action = f.get('action', '?')
                            text = f.get('text', '')[:60]
                            print(f"    [{action}] {text}...")
                else:
                    print(f"Entity '{entity_name}' not found")
                print()
                continue
            
            else:
                print(f"Unknown command: {cmd}")
                print("Type /help for available commands")
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
                    print(f"[DEBUG] Frame: agent={frame.get('agent')}, "
                          f"action={frame.get('action')}, patient={frame.get('patient')}")
        else:
            print("\nBot: I don't have information about that.")
        
        print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
