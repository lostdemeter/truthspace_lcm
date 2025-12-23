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
from .core.conversation_memory import ConversationMemory
from .core.reasoning_engine import ReasoningEngine
from .core.holographic_generator import HolographicGenerator


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
    parser.add_argument(
        '--certainty', '-w', type=float, default=0.0,
        help='Certainty dial: -1 (definitive) to +1 (hedged)'
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
    qa = ConceptQA(style_x=args.style, perspective_y=args.perspective, depth_z=args.depth, certainty_w=args.certainty)
    count = qa.load_corpus(str(corpus_path))
    print(f"Loaded {count} concept frames")
    
    # Train the model with quality examples
    from .training_data import train_model
    train_model(qa, verbose=True)
    
    # Initialize conversation memory
    memory = ConversationMemory(max_turns=10)
    
    # Initialize reasoning engine and holographic generator
    reasoning = ReasoningEngine(qa.knowledge)
    hologen = HolographicGenerator(qa.knowledge)
    
    # Show dial settings
    style_label = 'formal' if args.style < -0.3 else ('casual' if args.style > 0.3 else 'neutral')
    perspective_label = 'subjective' if args.perspective < -0.3 else ('meta' if args.perspective > 0.3 else 'objective')
    depth_label = 'terse' if args.depth < -0.3 else ('elaborate' if args.depth > 0.3 else 'standard')
    certainty_label = 'definitive' if args.certainty < -0.3 else ('hedged' if args.certainty > 0.3 else 'neutral')
    print(f"φ-Dial: style={style_label} (x={args.style:+.1f}), perspective={perspective_label} (y={args.perspective:+.1f})")
    print(f"        depth={depth_label} (z={args.depth:+.1f}), certainty={certainty_label} (w={args.certainty:+.1f})")
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
    print("  /certainty W - Set certainty dial (-1=definitive, +1=hedged)")
    print("  /dial       - Show current dial settings")
    print("  /help       - Show this help")
    print("  /quit       - Exit")
    print()
    print("Ask questions like:")
    print('  "Who is Darcy?"')
    print('  "What did Holmes do?"')
    print('  "Where is Netherfield?"')
    print('  "What did he do?" (uses conversation memory)')
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
                print("  /certainty W - Set certainty dial (-1=definitive, +1=hedged)")
                print("  /dial       - Show current dial settings")
                print("  /learn E T  - Teach: entity E should have answer T")
                print("  /learned    - Show learned structure")
                print("  /memory     - Show conversation memory")
                print("  /clear      - Clear conversation memory")
                print("  /why Q      - Multi-hop reasoning for WHY question")
                print("  /how Q      - Multi-hop reasoning for HOW question")
                print("  /path A B   - Find relationship path between A and B")
                print("  /holo E     - Holographic generation for entity E")
                print("  /quit       - Exit")
                print()
                continue
            
            elif cmd == '/dial':
                style_label = 'formal' if qa.style_x < -0.3 else ('casual' if qa.style_x > 0.3 else 'neutral')
                perspective_label = 'subjective' if qa.perspective_y < -0.3 else ('meta' if qa.perspective_y > 0.3 else 'objective')
                depth_label = 'terse' if qa.depth_z < -0.3 else ('elaborate' if qa.depth_z > 0.3 else 'standard')
                certainty_label = 'definitive' if qa.certainty_w < -0.3 else ('hedged' if qa.certainty_w > 0.3 else 'neutral')
                print(f"\nφ-Dial Settings (4D Quaternion):")
                print(f"  Style (x):       {qa.style_x:+.1f} ({style_label})")
                print(f"  Perspective (y): {qa.perspective_y:+.1f} ({perspective_label})")
                print(f"  Depth (z):       {qa.depth_z:+.1f} ({depth_label})")
                print(f"  Certainty (w):   {qa.certainty_w:+.1f} ({certainty_label})")
                print(f"  Label: {qa.projector.answer_generator.dial.get_full_label()}")
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
            
            elif cmd.startswith('/certainty'):
                parts = user_input.split()
                if len(parts) < 2:
                    print("Usage: /certainty <value>  (e.g., /certainty -1 for definitive, /certainty 1 for hedged)")
                    continue
                try:
                    w = float(parts[1])
                    qa.set_certainty(w)
                    certainty_label = 'definitive' if w < -0.3 else ('hedged' if w > 0.3 else 'neutral')
                    print(f"Certainty set to {w:+.1f} ({certainty_label})")
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
            
            elif cmd.startswith('/learn'):
                # Learn from a correction: /learn <entity> <correct answer>
                parts = user_input.split(maxsplit=2)
                if len(parts) < 3:
                    print("Usage: /learn <entity> <correct answer>")
                    print("Example: /learn holmes Holmes is a brilliant detective who investigates with Watson.")
                    continue
                
                entity = parts[1].lower()
                target = parts[2]
                
                # Initialize learnable if needed
                if qa.projector.answer_generator.learnable is None:
                    known = list(qa.knowledge.entities.keys())[:100]
                    qa.projector.answer_generator.init_learnable(known)
                
                # Get source for this entity
                entity_info = qa.knowledge.entities.get(entity, {})
                source = entity_info.get('source', 'the story') if isinstance(entity_info, dict) else 'the story'
                
                # Learn
                learned = qa.projector.answer_generator.learn_from_correction(entity, target, source)
                
                if learned:
                    print(f"Learned from correction:")
                    for item in learned:
                        print(f"  + {item}")
                    
                    # Show updated generation
                    new_answer = qa.projector.answer_generator.generate_from_learned(entity, source)
                    print(f"\nNew answer: \"{new_answer}\"")
                else:
                    print("Nothing new to learn from this correction.")
                print()
                continue
            
            elif cmd == '/learned':
                # Show learned structure
                if qa.projector.answer_generator.learnable is None:
                    print("No learned structure yet. Use /learn to teach me.")
                else:
                    stats = qa.projector.answer_generator.learnable.get_stats()
                    print(f"\nLearned Structure:")
                    print(f"  Entities with profiles: {stats['entities']}")
                    print(f"  Roles learned: {stats['roles_learned']}")
                    print(f"  Qualities learned: {stats['qualities_learned']}")
                    print(f"  Actions learned: {stats['actions_learned']}")
                    print(f"  Relations learned: {stats['relations_learned']}")
                    
                    # Show profiles
                    if stats['entities'] > 0:
                        print(f"\nProfiles:")
                        for entity, profile in qa.projector.answer_generator.learnable.profiles.items():
                            if profile.role or profile.qualities:
                                print(f"  {entity}: role={profile.role}, qualities={profile.qualities}, "
                                      f"actions={profile.actions}, relations={profile.relations}")
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
            
            elif cmd == '/memory':
                # Show conversation memory
                if memory:
                    print(f"\n{memory.get_summary()}")
                else:
                    print("\nNo conversation memory yet.")
                print()
                continue
            
            elif cmd == '/clear':
                # Clear conversation memory
                memory.clear()
                print("Conversation memory cleared.")
                print()
                continue
            
            elif cmd == '/why':
                # Multi-hop reasoning for WHY
                rest = user_input[4:].strip()
                if not rest:
                    print("Usage: /why <question>")
                else:
                    path = reasoning.reason(f"Why {rest}")
                    print(f"\n{path.answer}")
                    if path.steps:
                        print("Reasoning chain:")
                        for step in path.steps:
                            print(f"  → {step}")
                print()
                continue
            
            elif cmd == '/how':
                # Multi-hop reasoning for HOW
                rest = user_input[4:].strip()
                if not rest:
                    print("Usage: /how <question>")
                else:
                    path = reasoning.reason(f"How {rest}")
                    print(f"\n{path.answer}")
                    if path.steps:
                        print("Reasoning chain:")
                        for step in path.steps:
                            print(f"  → {step}")
                print()
                continue
            
            elif cmd == '/path':
                # Find relationship path between entities
                parts = user_input[5:].strip().split()
                if len(parts) < 2:
                    print("Usage: /path <entity1> <entity2>")
                else:
                    entity1, entity2 = parts[0], parts[1]
                    path = reasoning.reason(f"What is the relationship between {entity1} and {entity2}?")
                    print(f"\n{path.answer}")
                    if path.steps:
                        print("Path:")
                        for step in path.steps:
                            print(f"  → {step}")
                print()
                continue
            
            elif cmd == '/holo':
                # Holographic generation
                entity = user_input[5:].strip().lower()
                if not entity:
                    print("Usage: /holo <entity>")
                else:
                    # Use learned structure if available
                    learnable = qa.projector.answer_generator.learnable
                    output = hologen.generate(f"Who is {entity}?", entity=entity, learnable=learnable)
                    print(f"\nHolographic: {output}")
                print()
                continue
            
            else:
                print(f"Unknown command: {cmd}")
                print("Type /help for available commands")
                continue
        
        # Resolve pronouns using conversation memory
        original_input = user_input
        if memory.has_pronoun(user_input) and memory.focus_entity:
            user_input = memory.resolve_pronouns(user_input)
            if debug_mode:
                print(f"[DEBUG] Pronoun resolved: '{original_input}' → '{user_input}'")
        
        # Process question
        result = qa.ask_detailed(user_input)
        
        if debug_mode:
            print(f"\n[DEBUG] Axis: {result['axis']}, Entity: {result['entity']}")
        
        if result['answers']:
            best = result['answers'][0]
            
            print(f"\nBot: {best['answer']}")
            
            # Add turn to conversation memory
            memory.add_turn(
                query=original_input,
                answer=best['answer'],
                entity=result.get('entity'),
                axis=result.get('axis')
            )
            
            if debug_mode:
                print(f"\n[DEBUG] Confidence: {best['confidence']:.2f}")
                print(f"[DEBUG] Source: {best['source']}")
                print(f"[DEBUG] Frames used: {best.get('frame_count', 1)}")
                print(f"[DEBUG] Focus entity: {memory.focus_entity}")
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
