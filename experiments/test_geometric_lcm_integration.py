#!/usr/bin/env python3
"""
Test Geometric LCM Integration with TruthSpace

This test demonstrates the full integration of:
1. GeometricLCM (dynamic learning)
2. TruthSpace Vocabulary (text encoding)
3. Natural language parsing
4. Multi-hop reasoning
5. Persistence
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core import (
    GeometricLCM, 
    Vocabulary, 
    KnowledgeBase,
    GeoFact
)


def test_basic_functionality():
    """Test basic tell/ask/analogy."""
    print("=" * 70)
    print("TEST 1: Basic Functionality")
    print("=" * 70)
    
    lcm = GeometricLCM(dim=256)
    
    # Tell facts
    statements = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Tokyo is the capital of Japan.",
        "Rome is the capital of Italy.",
    ]
    
    print("\nLearning facts:")
    for s in statements:
        result = lcm.tell(s)
        print(f"  {result}")
    
    print(f"\nStatus: {lcm.status()}")
    
    # Ask questions
    print("\nAsking questions:")
    questions = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Japan?",
    ]
    
    for q in questions:
        answer = lcm.ask(q)
        print(f"  Q: {q}")
        print(f"  A: {answer}")
    
    # Test analogies
    print("\nTesting analogies:")
    analogies = [
        ("france", "paris", "germany", "berlin"),
        ("france", "paris", "japan", "tokyo"),
        ("germany", "berlin", "italy", "rome"),
    ]
    
    correct = 0
    for a, b, c, expected in analogies:
        results = lcm.analogy(a, b, c, k=1)
        answer = results[0][0] if results else "?"
        match = "✓" if answer == expected else "✗"
        print(f"  {match} {a}:{b} :: {c}:? → {answer}")
        if answer == expected:
            correct += 1
    
    print(f"\nAnalogy accuracy: {correct}/{len(analogies)} = {correct/len(analogies):.1%}")
    
    return correct == len(analogies)


def test_multi_domain():
    """Test learning across multiple domains."""
    print("\n" + "=" * 70)
    print("TEST 2: Multi-Domain Learning")
    print("=" * 70)
    
    lcm = GeometricLCM(dim=256)
    
    # Multiple domains
    facts = [
        # Geography
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Tokyo is the capital of Japan.",
        
        # Literature
        "Melville wrote Moby Dick.",
        "Shakespeare wrote Hamlet.",
        "Orwell wrote 1984.",
        
        # Locations
        "The Eiffel Tower is in Paris.",
        "The Colosseum is in Rome.",
    ]
    
    print("\nLearning multi-domain facts:")
    for f in facts:
        lcm.tell(f)
    
    print(f"Status: {lcm.status()}")
    
    # Test each domain
    print("\nGeography queries:")
    for country in ["france", "germany", "japan"]:
        results = lcm.query(country, "capital_of", k=1)
        if results:
            print(f"  {country} → {results[0][0]}")
    
    print("\nLiterature queries:")
    for author in ["melville", "shakespeare", "orwell"]:
        results = lcm.query(author, "wrote", k=1)
        if results:
            print(f"  {author} → {results[0][0]}")
    
    print("\nLocation queries:")
    for landmark in ["eiffel_tower", "colosseum"]:
        results = lcm.query(landmark, "located_in", k=1)
        if results:
            print(f"  {landmark} → {results[0][0]}")
    
    # Cross-domain analogies
    print("\nCross-domain analogies:")
    
    # Within geography
    results = lcm.analogy("france", "paris", "germany", k=1)
    print(f"  france:paris :: germany:? → {results[0][0] if results else '?'}")
    
    # Within literature
    results = lcm.analogy("melville", "moby_dick", "shakespeare", k=1)
    print(f"  melville:moby_dick :: shakespeare:? → {results[0][0] if results else '?'}")
    
    return True


def test_multi_hop_reasoning():
    """Test multi-hop reasoning."""
    print("\n" + "=" * 70)
    print("TEST 3: Multi-Hop Reasoning")
    print("=" * 70)
    
    lcm = GeometricLCM(dim=256)
    
    # Build knowledge graph
    facts = [
        "Paris is the capital of France.",
        "France is in Europe.",
        "The Eiffel Tower is in Paris.",
        
        "Berlin is the capital of Germany.",
        "Germany is in Europe.",
        
        "Tokyo is the capital of Japan.",
        "Japan is in Asia.",
    ]
    
    print("\nBuilding knowledge graph:")
    for f in facts:
        lcm.tell(f)
    
    print(f"Status: {lcm.status()}")
    
    # Multi-hop queries
    print("\nMulti-hop queries:")
    
    # france --capital_of--> ? --located_in--> ?
    print("\n  france --capital_of--> ? --located_in--> ?")
    results = lcm.multi_hop("france", ["capital_of", "located_in"], k=3)
    for entity, conf, path in results[:3]:
        print(f"    {' '.join(path)} (conf: {conf:.3f})")
    
    # eiffel_tower --located_in--> ? --capital_of--> ? (inverse)
    print("\n  Path: eiffel_tower → france")
    paths = lcm.find_path("eiffel_tower", "france", max_hops=2, k=3)
    for path, conf in paths[:3]:
        print(f"    {' → '.join(path)} (conf: {conf:.3f})")
    
    return True


def test_incremental_learning():
    """Test incremental learning."""
    print("\n" + "=" * 70)
    print("TEST 4: Incremental Learning")
    print("=" * 70)
    
    lcm = GeometricLCM(dim=256)
    
    # Initial facts
    print("\nPhase 1: Initial learning")
    initial = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
    ]
    for f in initial:
        lcm.tell(f)
    
    # Test
    results = lcm.analogy("france", "paris", "germany", k=1)
    print(f"  france:paris :: germany:? → {results[0][0] if results else '?'}")
    
    # Add more facts
    print("\nPhase 2: Adding more facts")
    more = [
        "Tokyo is the capital of Japan.",
        "Rome is the capital of Italy.",
        "Madrid is the capital of Spain.",
    ]
    for f in more:
        lcm.tell(f)
    
    # Test new analogies
    print("\nTesting new analogies:")
    new_analogies = [
        ("france", "paris", "japan", "tokyo"),
        ("france", "paris", "italy", "rome"),
        ("france", "paris", "spain", "madrid"),
    ]
    
    correct = 0
    for a, b, c, expected in new_analogies:
        results = lcm.analogy(a, b, c, k=1)
        answer = results[0][0] if results else "?"
        match = "✓" if answer == expected else "✗"
        print(f"  {match} {a}:{b} :: {c}:? → {answer}")
        if answer == expected:
            correct += 1
    
    print(f"\nNew analogy accuracy: {correct}/{len(new_analogies)} = {correct/len(new_analogies):.1%}")
    
    return correct == len(new_analogies)


def test_persistence():
    """Test save/load."""
    print("\n" + "=" * 70)
    print("TEST 5: Persistence")
    print("=" * 70)
    
    import tempfile
    import os
    
    lcm1 = GeometricLCM(dim=256)
    
    # Learn facts
    facts = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Melville wrote Moby Dick.",
    ]
    for f in facts:
        lcm1.tell(f)
    
    print(f"\nOriginal LCM: {lcm1.status()}")
    
    # Save
    filepath = tempfile.mktemp(suffix=".json")
    lcm1.save(filepath)
    print(f"Saved to: {filepath}")
    
    # Load into new instance
    lcm2 = GeometricLCM()
    lcm2.load(filepath)
    print(f"Loaded LCM: {lcm2.status()}")
    
    # Verify
    print("\nVerification:")
    answer = lcm2.ask("What is the capital of France?")
    print(f"  Q: What is the capital of France?")
    print(f"  A: {answer}")
    
    results = lcm2.analogy("france", "paris", "germany", k=1)
    print(f"  france:paris :: germany:? → {results[0][0] if results else '?'}")
    
    # Cleanup
    os.remove(filepath)
    
    return "paris" in answer.lower()


def test_integration_with_knowledge_base():
    """Test integration with existing KnowledgeBase."""
    print("\n" + "=" * 70)
    print("TEST 6: Integration with KnowledgeBase")
    print("=" * 70)
    
    # Create shared vocabulary
    vocab = Vocabulary(dim=256)
    
    # Create both systems
    kb = KnowledgeBase(vocab)
    lcm = GeometricLCM(dim=256, vocab=vocab)
    
    # Add facts to both
    text = """
    Paris is the capital of France.
    Berlin is the capital of Germany.
    Captain Ahab is the captain of the Pequod.
    Melville wrote Moby Dick.
    """
    
    # KnowledgeBase ingestion
    kb.ingest_text(text)
    print(f"\nKnowledgeBase: {len(kb.facts)} facts, {len(kb.qa_pairs)} Q&A pairs")
    
    # GeometricLCM ingestion
    for sentence in text.strip().split('\n'):
        sentence = sentence.strip()
        if sentence:
            lcm.ingest(sentence)
    lcm.learn(verbose=False)
    print(f"GeometricLCM: {lcm.status()}")
    
    # Query both
    print("\nQuerying both systems:")
    
    # KB search
    kb_results = kb.search_qa("What is the capital of France?", k=1)
    if kb_results:
        print(f"  KB: {kb_results[0][0].answer}")
    
    # LCM query
    lcm_answer = lcm.ask("What is the capital of France?")
    print(f"  LCM: {lcm_answer}")
    
    # LCM can do analogies that KB cannot
    print("\nAnalogies (LCM only):")
    results = lcm.analogy("france", "paris", "germany", k=1)
    print(f"  france:paris :: germany:? → {results[0][0] if results else '?'}")
    
    return True


def test_scaling():
    """Test scaling with larger data."""
    print("\n" + "=" * 70)
    print("TEST 7: Scaling")
    print("=" * 70)
    
    import time
    
    lcm = GeometricLCM(dim=256)
    
    # Generate synthetic data
    n_countries = 100
    n_authors = 50
    
    print(f"\nGenerating {n_countries} countries + {n_authors} authors...")
    
    for i in range(n_countries):
        lcm.add_fact(f"country_{i}", "capital_of", f"capital_{i}")
    
    for i in range(n_authors):
        lcm.add_fact(f"author_{i}", "wrote", f"book_{i}")
    
    print(f"Total facts: {len(lcm.facts)}")
    
    # Time learning
    start = time.time()
    consistency = lcm.learn(n_iterations=100, target_consistency=0.95, verbose=False)
    elapsed = time.time() - start
    
    print(f"Learning time: {elapsed:.3f}s")
    print(f"Final consistency: {consistency:.3f}")
    
    # Test accuracy
    correct = 0
    total = 20
    
    for i in range(total):
        results = lcm.query(f"country_{i}", "capital_of", k=1)
        if results and results[0][0] == f"capital_{i}":
            correct += 1
    
    print(f"Query accuracy: {correct}/{total} = {correct/total:.1%}")
    
    # Test analogies
    correct = 0
    total = 10
    
    for i in range(total):
        results = lcm.analogy("country_0", "capital_0", f"country_{i+1}", k=1)
        if results and results[0][0] == f"capital_{i+1}":
            correct += 1
    
    print(f"Analogy accuracy: {correct}/{total} = {correct/total:.1%}")
    
    return True


def main():
    """Run all tests."""
    print()
    print("GEOMETRIC LCM INTEGRATION TESTS")
    print("=" * 70)
    print()
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Multi-Domain", test_multi_domain),
        ("Multi-Hop Reasoning", test_multi_hop_reasoning),
        ("Incremental Learning", test_incremental_learning),
        ("Persistence", test_persistence),
        ("KB Integration", test_integration_with_knowledge_base),
        ("Scaling", test_scaling),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    passed_count = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed_count}/{len(results)} tests passed")
    
    return all(p for _, p in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
