"""
Tests for StackedLCM - 128D Hierarchical Geometric Embeddings

Tests core functionality:
- Layer encoding (morphological, lexical, syntactic, etc.)
- Embedding generation
- Clustering
- Resolution accuracy
- Intent detection (via chat interface)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm.core import StackedLCM, PHI
from truthspace_lcm.chat import LCMChat, Intent
import numpy as np


def test_initialization():
    """Test StackedLCM initialization."""
    print("Testing initialization...")
    
    lcm = StackedLCM()
    
    assert lcm.embedding_dim == 128, f"Expected 128D, got {lcm.embedding_dim}"
    assert len(lcm.layer_weights) == 7, f"Expected 7 layers, got {len(lcm.layer_weights)}"
    
    expected_layers = ['morphological', 'lexical', 'syntactic', 'compositional', 
                       'disambiguation', 'contextual', 'global']
    for layer in expected_layers:
        assert layer in lcm.layer_weights, f"Missing layer: {layer}"
    
    print("  ✅ Initialization tests passed")


def test_encoding():
    """Test embedding generation."""
    print("Testing encoding...")
    
    lcm = StackedLCM()
    
    # Test basic encoding
    emb = lcm.encode("list files in directory", update_stats=False)
    assert len(emb) == 128, f"Expected 128D embedding, got {len(emb)}"
    assert np.sum(emb) > 0, "Embedding should have non-zero values"
    
    # Test different texts produce different embeddings
    emb1 = lcm.encode("chop onions finely", update_stats=False)
    emb2 = lcm.encode("delete all files", update_stats=False)
    
    sim = lcm.cosine_similarity(emb1, emb2)
    assert sim < 0.8, f"Different domains should have low similarity, got {sim}"
    
    # Test similar texts produce similar embeddings
    emb3 = lcm.encode("slice carrots thinly", update_stats=False)
    sim_same = lcm.cosine_similarity(emb1, emb3)
    assert sim_same > sim, f"Same domain should have higher similarity"
    
    print("  ✅ Encoding tests passed")


def test_disambiguation():
    """Test context-dependent disambiguation."""
    print("Testing disambiguation...")
    
    lcm = StackedLCM()
    
    # These should be different despite sharing "cut"
    emb_tech = lcm.encode("cut the file", update_stats=False)
    emb_cook = lcm.encode("cut the vegetables", update_stats=False)
    
    sim = lcm.cosine_similarity(emb_tech, emb_cook)
    assert sim < 0.6, f"Ambiguous words should be disambiguated, got {sim}"
    
    # Search disambiguation
    emb_search_tech = lcm.encode("search for files", update_stats=False)
    emb_search_cook = lcm.encode("search for recipes", update_stats=False)
    
    sim_search = lcm.cosine_similarity(emb_search_tech, emb_search_cook)
    assert sim_search < 0.6, f"Search disambiguation failed, got {sim_search}"
    
    print("  ✅ Disambiguation tests passed")


def test_ingestion_and_clustering():
    """Test knowledge ingestion and clustering."""
    print("Testing ingestion and clustering...")
    
    lcm = StackedLCM()
    
    # Ingest mixed knowledge
    knowledge = [
        {"content": "ls -la", "description": "list files directory terminal"},
        {"content": "cat file.txt", "description": "show file contents terminal"},
        {"content": "chop onions", "description": "cut vegetables cooking food"},
        {"content": "boil water", "description": "heat water cooking stove"},
        {"content": "hello", "description": "greeting welcome hi"},
    ]
    
    lcm.ingest_batch(knowledge)
    
    assert len(lcm.points) == 5, f"Expected 5 points, got {len(lcm.points)}"
    assert len(lcm.clusters) > 0, "Should have at least one cluster"
    assert len(lcm.clusters) <= 5, "Should not have more clusters than points"
    
    print(f"  Created {len(lcm.clusters)} clusters from 5 points")
    print("  ✅ Ingestion and clustering tests passed")


def test_resolution():
    """Test query resolution."""
    print("Testing resolution...")
    
    lcm = StackedLCM()
    
    # Bootstrap with knowledge - use distinctive descriptions
    knowledge = [
        {"content": "ls -la", "description": "list show the files in directory folder terminal bash command"},
        {"content": "ps aux", "description": "show running processes system process terminal bash command"},
        {"content": "df -h", "description": "disk space storage usage free terminal bash command"},
        {"content": "chop onions", "description": "cut chop the vegetables onions cooking food kitchen knife"},
    ]
    lcm.ingest_batch(knowledge)
    
    # Test resolution - use queries that match the descriptions well
    test_cases = [
        ("show the files in directory", "ls -la"),
        ("disk space storage", "df -h"),
        ("chop the vegetables", "chop onions"),
    ]
    
    passed = 0
    for query, expected in test_cases:
        content, sim, cluster = lcm.resolve(query)
        if content == expected:
            passed += 1
        else:
            print(f"    ✗ '{query}' → {content} (expected {expected}, sim={sim:.2f})")
    
    accuracy = passed / len(test_cases)
    assert accuracy >= 0.6, f"Accuracy {accuracy:.0%} below threshold"
    
    print(f"  ✅ Resolution tests passed ({passed}/{len(test_cases)})")


def test_chat_intent_detection():
    """Test chat interface intent detection."""
    print("Testing intent detection...")
    
    chat = LCMChat(safe_mode=True)
    
    test_cases = [
        ("hello", Intent.CHAT),
        ("list files", Intent.BASH),
        ("ls -la", Intent.BASH),
        ("exit", Intent.EXIT),
        ("help", Intent.HELP),
        ("how do I find a file?", Intent.QUESTION),
        ("show running processes", Intent.BASH),
    ]
    
    passed = 0
    for query, expected_intent in test_cases:
        intent, conf = chat.detect_intent(query)
        if intent == expected_intent:
            passed += 1
        else:
            print(f"    ✗ '{query}' → {intent} (expected {expected_intent})")
    
    accuracy = passed / len(test_cases)
    assert accuracy >= 0.7, f"Intent detection accuracy {accuracy:.0%} below threshold"
    
    print(f"  ✅ Intent detection tests passed ({passed}/{len(test_cases)})")


def test_bash_resolution():
    """Test natural language to bash command resolution."""
    print("Testing bash resolution...")
    
    chat = LCMChat(safe_mode=True)
    
    test_cases = [
        ("list files", "ls -la"),
        ("show disk space", "df -h"),
        ("show running processes", "ps aux"),
        ("git status", "git status"),
    ]
    
    passed = 0
    for query, expected in test_cases:
        cmd, conf = chat.resolve_bash_command(query)
        if cmd == expected:
            passed += 1
        else:
            print(f"    ✗ '{query}' → {cmd} (expected {expected})")
    
    accuracy = passed / len(test_cases)
    assert accuracy >= 0.75, f"Bash resolution accuracy {accuracy:.0%} below threshold"
    
    print(f"  ✅ Bash resolution tests passed ({passed}/{len(test_cases)})")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("StackedLCM Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_initialization,
        test_encoding,
        test_disambiguation,
        test_ingestion_and_clustering,
        test_resolution,
        test_chat_intent_detection,
        test_bash_resolution,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} FAILED: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
