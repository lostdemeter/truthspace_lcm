#!/usr/bin/env python3
"""
Interactive Demo for Geometric Chatbot

This script provides an interactive demonstration of the geometric chatbot,
allowing users to explore its capabilities and understand how it works.

Usage:
    python demo.py              # Run interactive demo
    python demo.py --tutorial   # Run step-by-step tutorial
    python demo.py --benchmark  # Run benchmark tests

Author: Lesley Gushurst
License: GPLv3
"""

from geometric_chatbot import (
    GeometricChatbot, 
    GeometricMorphology, 
    GeometricConjugation,
    MORPHOLOGY_BOOTSTRAP,
    EXAMPLE_CORPUS
)


def tutorial():
    """Step-by-step tutorial explaining each geometric component."""
    
    print("=" * 70)
    print("GEOMETRIC CHATBOT TUTORIAL")
    print("=" * 70)
    print()
    
    # Step 1: Morphology Bootstrap
    print("STEP 1: GEOMETRIC MORPHOLOGY")
    print("-" * 70)
    print("""
The first step is learning morphological equivalence from parallel structures.

We use sentences like:
    "I love. He loves. I loved."
    
This teaches the system that 'love', 'loves', and 'loved' are the same
concept at different temporal phases:
    - Position 0: base form (love)
    - Position 1: 3rd person singular (loves)
    - Position 2: past tense (loved)
""")
    
    morph = GeometricMorphology()
    morph.bootstrap(MORPHOLOGY_BOOTSTRAP)
    
    print("Morphological equivalences learned:")
    test_words = ['love', 'loves', 'loved', 'go', 'goes', 'went', 'think', 'thinks', 'thought']
    for word in test_words:
        equivalents = morph.get_equivalents(word)
        if len(equivalents) > 1:
            print(f"  {word} ≡ {', '.join(sorted(equivalents - {word}))}")
    
    input("\nPress Enter to continue...")
    print()
    
    # Step 2: Conjugation
    print("STEP 2: GEOMETRIC CONJUGATION")
    print("-" * 70)
    print("""
Using the same bootstrap, we learn to conjugate verbs.

Given a word and a target phase, we can produce the correct form:
    - Phase 0: base form
    - Phase 1: 3rd person singular
    - Phase 2: past tense
""")
    
    conj = GeometricConjugation()
    conj.bootstrap(MORPHOLOGY_BOOTSTRAP)
    
    print("Conjugation examples:")
    test_cases = [
        ('love', 0, 'base'),
        ('love', 1, '3rd singular'),
        ('love', 2, 'past'),
        ('go', 1, '3rd singular'),
        ('go', 2, 'past'),
        ('think', 2, 'past'),
    ]
    for word, phase, desc in test_cases:
        result = conj.conjugate(word, phase)
        print(f"  {word} → phase {phase} ({desc}): {result}")
    
    input("\nPress Enter to continue...")
    print()
    
    # Step 3: Frame Extraction
    print("STEP 3: POSITION-BASED FRAME EXTRACTION")
    print("-" * 70)
    print("""
We extract semantic frames using position bands:

    Position [0.0, 0.33)  → Initiator (subject)
    Position [0.33, 0.66) → Mediator (verb)
    Position [0.66, 1.0]  → Receiver (object)

Example: "Holmes examined the evidence carefully"
    - Position 0.0: Holmes → Initiator
    - Position 0.2: examined → Mediator
    - Position 0.6: evidence → Receiver
""")
    
    bot = GeometricChatbot()
    bot.learn(EXAMPLE_CORPUS)
    
    print("Sample frames extracted:")
    for frame in bot.frames[:10]:
        print(f"  {frame.initiator} → {frame.mediator} → {frame.receiver or '∅'}")
    
    input("\nPress Enter to continue...")
    print()
    
    # Step 4: Stop Word Detection
    print("STEP 4: GEOMETRIC STOP WORD DETECTION")
    print("-" * 70)
    print("""
Stop words are detected geometrically based on semantic role absence.

A word is a stop word if:
    1. It has no semantic role (never initiator, mediator, or receiver)
    2. OR: It's short (≤4 chars) and frequent (≥3 occurrences)
    3. OR: It only appears as receiver and is short (catches prepositions)

No hard-coded list needed!
""")
    
    stop_words = sorted([n for n, c in bot.concepts.items() if c.is_geometric_stop_word])
    content_words = sorted([n for n, c in bot.concepts.items() if c.is_content_word])
    
    print(f"Geometrically detected stop words ({len(stop_words)}):")
    print(f"  {', '.join(stop_words[:15])}...")
    print()
    print(f"Content words ({len(content_words)}):")
    print(f"  {', '.join(content_words[:15])}...")
    
    input("\nPress Enter to continue...")
    print()
    
    # Step 5: Query Processing
    print("STEP 5: QUERY PROCESSING")
    print("-" * 70)
    print("""
When processing a query, we:
    1. Tokenize and find content words
    2. Use morphological equivalence to match concepts
    3. Detect question type geometrically
    4. Generate response using geometric conjugation
""")
    
    queries = [
        "Who is Holmes?",
        "Who killed?",
        "Who loves?",
        "What does Watson do?",
    ]
    
    print("Query examples:")
    for q in queries:
        print(f"\n  Q: {q}")
        print(f"  A: {bot.respond(q)}")
    
    print()
    print("=" * 70)
    print("TUTORIAL COMPLETE")
    print("=" * 70)
    print("""
Key takeaways:
    • No hard-coded stop word lists
    • No suffix rules for morphology
    • No part-of-speech tagging
    • Everything is learned from position and parallel structure
""")


def benchmark():
    """Run benchmark tests on the geometric chatbot."""
    
    print("=" * 70)
    print("GEOMETRIC CHATBOT BENCHMARK")
    print("=" * 70)
    print()
    
    bot = GeometricChatbot()
    bot.learn(EXAMPLE_CORPUS)
    
    # Test cases with expected answers
    test_cases = [
        ("Who is Holmes?", "Holmes", "protagonist"),
        ("Who killed?", "Hamlet", "kills"),
        ("Who loves?", "Ophelia", "loves"),
        ("What does Watson do?", "Watson", "watches"),
        ("Tell me about Alice", "Alice", "protagonist"),
        ("Who examined?", "Holmes", "examines"),
        ("Who fell?", "Alice", "falls"),
        ("Who poisoned?", "Claudius", "poisons"),
    ]
    
    print("Running benchmark tests...")
    print("-" * 70)
    
    passed = 0
    failed = 0
    
    for query, expected_subject, expected_content in test_cases:
        response = bot.respond(query)
        
        # Check if expected content is in response
        subject_match = expected_subject.lower() in response.lower()
        content_match = expected_content.lower() in response.lower()
        
        if subject_match and content_match:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        
        print(f"{status}: {query}")
        print(f"       Response: {response}")
        print(f"       Expected: {expected_subject}, {expected_content}")
        print()
    
    print("-" * 70)
    print(f"Results: {passed}/{len(test_cases)} passed ({100*passed/len(test_cases):.0f}%)")
    print()
    
    # Morphology benchmark
    print("MORPHOLOGY BENCHMARK")
    print("-" * 70)
    
    morph_tests = [
        ('love', 'loved', True),
        ('love', 'loves', True),
        ('go', 'went', True),
        ('think', 'thought', True),
        ('love', 'hate', False),
        ('run', 'walk', False),
    ]
    
    morph_passed = 0
    for word1, word2, expected in morph_tests:
        result = bot.morphology.are_equivalent(word1, word2)
        if result == expected:
            status = "✓"
            morph_passed += 1
        else:
            status = "✗"
        print(f"  {status} {word1} ≡ {word2}? {result} (expected: {expected})")
    
    print(f"\nMorphology: {morph_passed}/{len(morph_tests)} passed")
    print()
    
    # Conjugation benchmark
    print("CONJUGATION BENCHMARK")
    print("-" * 70)
    
    conj_tests = [
        ('love', 0, 'love'),
        ('love', 1, 'loves'),
        ('love', 2, 'loved'),
        ('go', 1, 'goes'),
        ('go', 2, 'went'),
        ('think', 2, 'thought'),
    ]
    
    conj_passed = 0
    for word, phase, expected in conj_tests:
        result = bot.conjugation.conjugate(word, phase)
        if result == expected:
            status = "✓"
            conj_passed += 1
        else:
            status = "✗"
        print(f"  {status} {word} → phase {phase}: {result} (expected: {expected})")
    
    print(f"\nConjugation: {conj_passed}/{len(conj_tests)} passed")


def interactive():
    """Run interactive chat session."""
    
    print("=" * 70)
    print("GEOMETRIC CHATBOT - Interactive Mode")
    print("=" * 70)
    print()
    
    bot = GeometricChatbot()
    bot.learn(EXAMPLE_CORPUS)
    
    print(f"Learned from {bot.total_sentences} sentences.")
    print()
    print("Commands:")
    print("  'quit'     - Exit")
    print("  'analysis' - Show word analysis")
    print("  'frames'   - Show extracted frames")
    print("  'help'     - Show this help")
    print()
    
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        cmd = query.lower()
        
        if cmd == 'quit':
            print("Goodbye!")
            break
        
        if cmd == 'help':
            print("Commands: quit, analysis, frames, help")
            print("Or ask a question like:")
            print("  - Who is Holmes?")
            print("  - Who killed?")
            print("  - What does Watson do?")
            continue
        
        if cmd == 'analysis':
            bot.show_analysis()
            continue
        
        if cmd == 'frames':
            print("\nExtracted frames:")
            for frame in bot.frames:
                print(f"  {frame.initiator} → {frame.mediator} → {frame.receiver or '∅'}")
            print()
            continue
        
        response = bot.respond(query)
        print(f"Bot: {response}")
        print()


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == '--tutorial':
            tutorial()
        elif arg == '--benchmark':
            benchmark()
        elif arg == '--interactive':
            interactive()
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python demo.py [--tutorial|--benchmark|--interactive]")
    else:
        interactive()


if __name__ == "__main__":
    main()
