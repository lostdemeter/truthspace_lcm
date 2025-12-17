"""
Tests for TruthSpace LCM - Hypergeometric Resolution

Tests core functionality:
- φ-MAX encoding
- Geometric distance
- Knowledge storage and retrieval
- Resolution accuracy
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm import TruthSpace, KnowledgeGapError, PHI


def test_encoding():
    """Test φ-MAX encoding."""
    print("Testing φ-MAX encoding...")
    
    ts = TruthSpace()
    
    # Test basic encoding
    pos = ts._encode("read file")
    assert pos[1] > 0, "READ should activate dim 1"
    assert pos[5] > 0, "FILE should activate dim 5"
    
    # Test synonym collapse (Sierpinski property)
    pos1 = ts._encode("storage")
    pos2 = ts._encode("disk storage")
    pos3 = ts._encode("disk space storage")
    
    assert abs(pos1[5] - pos2[5]) < 0.001, "Synonyms should collapse"
    assert abs(pos2[5] - pos3[5]) < 0.001, "Synonyms should collapse"
    
    print("  ✅ Encoding tests passed")


def test_resolution():
    """Test geometric resolution."""
    print("Testing resolution...")
    
    ts = TruthSpace()
    
    test_cases = [
        ("list directory contents", "ls"),
        ("show disk space", "df"),
        ("copy files", "cp"),
        ("move files", "mv"),
        ("find files", "find"),
        ("search text in files", "grep"),
        ("show running processes", "ps"),
        ("kill process", "kill"),
        ("compress files", "tar"),
    ]
    
    passed = 0
    for query, expected in test_cases:
        output, entry, sim = ts.resolve(query)
        if output == expected:
            passed += 1
        else:
            print(f"    ✗ '{query}' → {output} (expected {expected})")
    
    accuracy = passed / len(test_cases)
    assert accuracy >= 0.8, f"Accuracy {accuracy:.0%} below threshold"
    
    print(f"  ✅ Resolution tests passed ({passed}/{len(test_cases)})")


def test_store_and_retrieve():
    """Test storing and retrieving knowledge."""
    print("Testing store/retrieve...")
    
    ts = TruthSpace()
    
    # Store new knowledge
    ts.store("kubectl apply", "deploy application")
    
    # Retrieve it
    output, entry, sim = ts.resolve("deploy application")
    assert output == "kubectl apply", f"Expected 'kubectl apply', got '{output}'"
    assert sim > 0.9, f"Similarity {sim} too low"
    
    print("  ✅ Store/retrieve tests passed")


def test_knowledge_gap():
    """Test KnowledgeGapError."""
    print("Testing knowledge gap...")
    
    ts = TruthSpace()
    
    # Query with very low threshold should fail for nonsense
    try:
        ts.query("xyzzy foobar baz", threshold=0.9)
        assert False, "Should have raised KnowledgeGapError"
    except KnowledgeGapError:
        pass
    
    print("  ✅ Knowledge gap tests passed")


def test_explain():
    """Test explanation."""
    print("Testing explain...")
    
    ts = TruthSpace()
    
    explanation = ts.explain("show disk space")
    assert "Query:" in explanation
    assert "φ-MAX Encoding:" in explanation
    assert "Top 3 matches:" in explanation
    
    print("  ✅ Explain tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TruthSpace LCM Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_encoding,
        test_resolution,
        test_store_and_retrieve,
        test_knowledge_gap,
        test_explain,
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
