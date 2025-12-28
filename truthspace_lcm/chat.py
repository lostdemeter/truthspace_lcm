#!/usr/bin/env python3
"""
TruthSpace LCM Unified Chat Interface

Fully Geometric Language Understanding with Holographic Templates and Semantic Quaternions.

Architecture:
    Question (any language)
            ↓
    Geometric Frame Extraction (position bands)
            ↓
    φ-Space Query (language-agnostic)
            ↓
    Holographic Template Projection + Semantic Quaternion
            ↓
    φ-Dial Styled Response

Key Features:
- No hard-coded stop word lists (geometric detection)
- No hard-coded verb mappings (learned from parallel structures)
- Position-based frame extraction
- Holographic template projection for dynamic responses
- Semantic quaternions for analogies (100% accuracy)
- Two quaternions: φ-dial (output) + semantic (encoding)

Usage:
    python -m truthspace_lcm.chat_unified
    python -m truthspace_lcm.chat_unified --debug
    python -m truthspace_lcm.chat_unified --corpus path/to/corpus.json
    python -m truthspace_lcm.chat_unified --style -1 --perspective 1  # Formal + Meta
"""

import argparse
import sys
from pathlib import Path

from .core import (
    HolographicGeometricQA,
    ConversationMemory,
    ReasoningEngine,
    CodeGenerator,
    Planner,
)


def main():
    parser = argparse.ArgumentParser(
        description='TruthSpace LCM - Geometric Language Model'
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
        if not corpus_path.exists():
            corpus_path = Path(__file__).parent / 'sample_corpus_geometric.json'
    
    # Initialize Q&A system
    print("=" * 60)
    print("  TruthSpace LCM - Geometric Language Model")
    print("  Holographic Templates + Semantic Quaternions")
    print("=" * 60)
    print()
    print("Architecture:")
    print("  • Geometric frame extraction (position bands)")
    print("  • Holographic template projection (dynamic responses)")
    print("  • Semantic quaternions (100% analogy accuracy)")
    print("  • φ-dial output styling (4D quaternion)")
    print()
    
    if not corpus_path.exists():
        print(f"Error: Corpus not found at {corpus_path}")
        print()
        print("To build a corpus, run:")
        print("  python scripts/build_concept_corpus.py")
        return 1
    
    print(f"Loading corpus from {corpus_path}...")
    qa = HolographicGeometricQA()
    qa.set_style(args.style)
    qa.set_perspective(args.perspective)
    count = qa.load_corpus(str(corpus_path))
    print(f"Loaded {count} concept frames")
    print(f"Total sentences learned: {qa.knowledge.total_sentences}")
    print(f"Total concepts: {len(qa.knowledge.concepts)}")
    print(f"Frames extracted: {len(qa.knowledge.frames)}")
    print(f"Morphology clusters: {len(qa.knowledge.morphology.equivalence_classes)}")
    
    # Show geometric analysis
    stop_words = [n for n, c in qa.knowledge.concepts.items() if c.is_geometric_stop_word]
    content_words = [n for n, c in qa.knowledge.concepts.items() if c.is_content_word]
    print(f"Geometric stop words: {len(stop_words)}")
    print(f"Content words: {len(content_words)}")
    
    # Initialize conversation memory
    memory = ConversationMemory(max_turns=10)
    
    # Initialize supporting components
    reasoning = ReasoningEngine(qa.knowledge)
    codegen = CodeGenerator()
    planner = Planner(codegen)
    
    # Show dial settings
    style_label = 'formal' if args.style < -0.3 else ('casual' if args.style > 0.3 else 'neutral')
    perspective_label = 'subjective' if args.perspective < -0.3 else ('meta' if args.perspective > 0.3 else 'objective')
    depth_label = 'terse' if args.depth < -0.3 else ('elaborate' if args.depth > 0.3 else 'standard')
    certainty_label = 'definitive' if args.certainty < -0.3 else ('hedged' if args.certainty > 0.3 else 'neutral')
    print(f"\nφ-Dial: style={style_label} (x={args.style:+.1f}), perspective={perspective_label} (y={args.perspective:+.1f})")
    print(f"        depth={depth_label} (z={args.depth:+.1f}), certainty={certainty_label} (w={args.certainty:+.1f})")
    print()
    
    # Show sample entities
    entities = list(qa.knowledge.concepts.keys())
    top_entities = sorted(
        [e for e in entities if qa.knowledge.concepts[e].is_content_word],
        key=lambda e: qa.knowledge.concepts[e].initiator_count + qa.knowledge.concepts[e].receiver_count,
        reverse=True
    )[:8]
    
    if top_entities:
        print("Sample concepts:")
        for e in top_entities:
            c = qa.knowledge.concepts[e]
            role = "initiator" if c.phi_direction > 0.3 else "receiver" if c.phi_direction < -0.3 else "neutral"
            print(f"  - {e.title()} ({role}, φ-dir={c.phi_direction:.2f})")
        print()
    
    print("Commands:")
    print("  /debug      - Toggle debug mode")
    print("  /concept X  - Show geometric info about concept X")
    print("  /stats      - Show corpus statistics")
    print("  /style X    - Set style dial (-1=formal, +1=casual)")
    print("  /perspective Y - Set perspective dial (-1=subjective, +1=meta)")
    print("  /analogy A B C - Complete analogy A:B::C:?")
    print("  /similar A B - Find pairs with similar relation to A:B")
    print("  /memory     - Show conversation memory")
    print("  /clear      - Clear conversation memory")
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
                print("  /concept X  - Show geometric info about concept X")
                print("  /stats      - Show corpus statistics")
                print("  /style X    - Set style dial (-1=formal, +1=casual)")
                print("  /perspective Y - Set perspective dial (-1=subjective, +1=meta)")
                print("  /analogy A B C - Complete analogy A:B::C:?")
                print("  /similar A B - Find pairs with similar relation to A:B")
                print("  /memory     - Show conversation memory")
                print("  /clear      - Clear conversation memory")
                print("  /code REQ   - Generate Python code from request")
                print("  /run TASK   - Plan and execute a task")
                print("  /quit       - Exit")
                print()
                continue
            
            elif cmd == '/stats':
                print(f"\nCorpus Statistics:")
                print(f"  Total frames: {len(qa.knowledge.frames)}")
                print(f"  Total concepts: {len(qa.knowledge.concepts)}")
                print(f"  Content words: {len([c for c in qa.knowledge.concepts.values() if c.is_content_word])}")
                print(f"  Stop words: {len([c for c in qa.knowledge.concepts.values() if c.is_geometric_stop_word])}")
                print(f"  Morphology clusters: {len(qa.knowledge.morphology.equivalence_classes)}")
                print(f"  Semantic quaternion concepts: {len(qa.semantic_navigator.concepts)}")
                print()
                continue
            
            elif cmd.startswith('/concept'):
                parts = user_input.split()
                if len(parts) < 2:
                    print("Usage: /concept <word>")
                    continue
                word = parts[1].lower()
                if word in qa.knowledge.concepts:
                    c = qa.knowledge.concepts[word]
                    print(f"\nConcept: {word}")
                    print(f"  φ-direction: {c.phi_direction:.3f}")
                    print(f"  Mean position: {c.mean_position:.3f}")
                    print(f"  Initiator count: {c.initiator_count}")
                    print(f"  Mediator count: {c.mediator_count}")
                    print(f"  Receiver count: {c.receiver_count}")
                    print(f"  Is content word: {c.is_content_word}")
                    print(f"  Is stop word: {c.is_geometric_stop_word}")
                    
                    # Show semantic quaternion if available
                    if word in qa.semantic_navigator.concepts:
                        sq = qa.semantic_navigator.concepts[word]
                        print(f"  Semantic quaternion: x={sq.x:.2f}, y={sq.y:.2f}, z={sq.z:.2f}, w={sq.w:.2f}")
                else:
                    print(f"Concept '{word}' not found in knowledge base")
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
                    print("Usage: /perspective <value>")
                    continue
                try:
                    y = float(parts[1])
                    qa.set_perspective(y)
                    perspective_label = 'subjective' if y < -0.3 else ('meta' if y > 0.3 else 'objective')
                    print(f"Perspective set to {y:+.1f} ({perspective_label})")
                except ValueError:
                    print("Invalid value. Use a number between -1 and 1.")
                continue
            
            elif cmd.startswith('/analogy'):
                parts = user_input.split()
                if len(parts) < 4:
                    print("Usage: /analogy A B C  (A:B::C:?)")
                    continue
                a, b, c = parts[1].lower(), parts[2].lower(), parts[3].lower()
                results = qa.complete_analogy(a, b, c, k=5)
                print(f"\n{a} : {b} :: {c} : ?")
                print(f"Top answers: {[r[0] for r in results[:5]]}")
                print()
                continue
            
            elif cmd.startswith('/similar'):
                parts = user_input.split()
                if len(parts) < 3:
                    print("Usage: /similar A B  (find pairs with similar relation)")
                    continue
                a, b = parts[1].lower(), parts[2].lower()
                results = qa.find_similar_relations(a, b, k=5)
                print(f"\nPairs with similar relation to '{a}' → '{b}':")
                for w1, w2, sim in results:
                    print(f"  {w1} → {w2}: {sim:.3f}")
                print()
                continue
            
            elif cmd == '/memory':
                if not memory.turns:
                    print("Conversation memory is empty")
                else:
                    print("\nConversation Memory:")
                    for i, turn in enumerate(memory.turns):
                        print(f"  {i+1}. Q: {turn.query[:50]}...")
                        print(f"     A: {turn.response[:50]}...")
                print()
                continue
            
            elif cmd == '/clear':
                memory.clear()
                print("Conversation memory cleared")
                continue
            
            elif cmd.startswith('/code'):
                request = user_input[5:].strip()
                if not request:
                    print("Usage: /code <request>")
                    continue
                code_frame = codegen.generate(request)
                if code_frame:
                    print(f"\n# {code_frame.description}")
                    print(code_frame.code)
                else:
                    print("Could not generate code for that request")
                print()
                continue
            
            elif cmd.startswith('/run'):
                task = user_input[4:].strip()
                if not task:
                    print("Usage: /run <task>")
                    continue
                plan = planner.plan(task)
                if plan:
                    print(f"\nPlan: {plan.goal}")
                    for i, step in enumerate(plan.steps):
                        print(f"  {i+1}. {step.description}")
                    print("\nExecuting...")
                    result = planner.execute(plan)
                    print(f"Result: {result}")
                else:
                    print("Could not create a plan for that task")
                print()
                continue
            
            else:
                print(f"Unknown command: {cmd}")
                print("Type /help for available commands")
                continue
        
        # Regular question - use holographic geometric QA
        # Resolve pronouns using conversation memory
        resolved_query = memory.resolve_pronouns(user_input)
        
        if debug_mode:
            print(f"\n[DEBUG] Original: {user_input}")
            print(f"[DEBUG] Resolved: {resolved_query}")
        
        # Get answer using holographic templates
        result = qa.ask_detailed(resolved_query)
        
        if debug_mode:
            print(f"[DEBUG] Axis: {result.get('axis', 'unknown')}")
            print(f"[DEBUG] Entity: {result.get('entity', 'unknown')}")
            if result.get('answers'):
                print(f"[DEBUG] Source: {result['answers'][0].get('source', 'unknown')}")
        
        if result.get('answers'):
            answer = result['answers'][0]['answer']
            print(f"\nLCM: {answer}")
            
            # Add to memory
            entities = result.get('entities', [])
            memory.add_turn(user_input, answer, entities)
        else:
            print("\nLCM: I don't have information about that.")
        
        print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
