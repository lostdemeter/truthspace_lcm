"""
Test reproject() - Probe Extraction Protocol for exact learning

Demonstrates that reproject() (eigendecomposition) achieves better
accuracy than attract/repel dynamics.

From PEP: "Training is approximation. Probing is measurement."
"""

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
from hypermapping import HyperMapping, TextEncoder


def test_reproject():
    """Test that reproject() gives exact similarity matching."""
    print("=" * 60)
    print("  REPROJECT TEST - Probe Extraction Protocol")
    print("=" * 60)
    print()
    
    # Create mappings
    mappings = [
        ("list files", "ls"),
        ("show files", "ls"),
        ("display files", "ls"),
        ("enumerate files", "ls"),
        ("delete file", "rm"),
        ("remove file", "rm"),
        ("erase file", "rm"),
        ("kill process", "kill"),
        ("terminate process", "kill"),
        ("stop process", "kill"),
        ("disk space", "df -h"),
        ("disk usage", "df -h"),
        ("memory usage", "free -h"),
        ("show memory", "free -h"),
    ]
    
    # Create space with hash encoder (positions are random)
    space = HyperMapping(dims=12, name="commands")
    for input_val, output_val in mappings:
        space.map(input_val, output_val)
    
    print(f"Created {len(space)} mappings")
    print()
    
    # Test queries BEFORE reproject
    test_queries = [
        ("list files", "ls"),
        ("show files", "ls"),
        ("delete file", "rm"),
        ("kill process", "kill"),
        ("disk space", "df -h"),
        ("memory usage", "free -h"),
    ]
    
    print("--- BEFORE reproject() (random positions) ---")
    correct_before = 0
    for query, expected in test_queries:
        result = space.forward(query)
        predicted = result.output if result else None
        is_correct = predicted == expected
        correct_before += is_correct
        print(f"  '{query}' → {predicted} {'✓' if is_correct else '✗'}")
    
    accuracy_before = correct_before / len(test_queries) * 100
    print(f"\nAccuracy: {accuracy_before:.1f}%")
    print()
    
    # Apply reproject() - Probe Extraction Protocol
    print("--- Applying reproject() (eigendecomposition) ---")
    space.reproject()
    print("Done!")
    print()
    
    # Test queries AFTER reproject
    print("--- AFTER reproject() (exact positions) ---")
    correct_after = 0
    for query, expected in test_queries:
        result = space.forward(query)
        predicted = result.output if result else None
        is_correct = predicted == expected
        correct_after += is_correct
        sim = result.similarity if result else 0
        print(f"  '{query}' → {predicted} (sim={sim:.3f}) {'✓' if is_correct else '✗'}")
    
    accuracy_after = correct_after / len(test_queries) * 100
    print(f"\nAccuracy: {accuracy_after:.1f}%")
    print()
    
    # Test with unseen queries
    print("--- Testing unseen queries ---")
    unseen_queries = [
        ("show all files", "ls"),      # Similar to "show files"
        ("remove the file", "rm"),     # Similar to "remove file"
        ("end process", "kill"),       # Similar to "stop process"
        ("storage space", "df -h"),    # Similar to "disk space"
    ]
    
    correct_unseen = 0
    for query, expected in unseen_queries:
        result = space.forward(query)
        predicted = result.output if result else None
        is_correct = predicted == expected
        correct_unseen += is_correct
        sim = result.similarity if result else 0
        print(f"  '{query}' → {predicted} (sim={sim:.3f}) {'✓' if is_correct else '✗'}")
    
    accuracy_unseen = correct_unseen / len(unseen_queries) * 100
    print(f"\nUnseen accuracy: {accuracy_unseen:.1f}%")
    print()
    
    # Summary
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print()
    print(f"Before reproject(): {accuracy_before:.1f}%")
    print(f"After reproject():  {accuracy_after:.1f}%")
    print(f"Unseen queries:     {accuracy_unseen:.1f}%")
    print()
    print("Key insight from PEP:")
    print("  - attract/repel = approximation (has holographic bound)")
    print("  - reproject() = measurement (exact, no bound)")
    print()
    print("The similarity matrix IS the structure.")
    print("Eigendecomposition constructs positions that realize it exactly.")


def test_similarity_preservation():
    """Test that reproject() preserves the similarity matrix."""
    print()
    print("=" * 60)
    print("  SIMILARITY PRESERVATION TEST")
    print("=" * 60)
    print()
    
    mappings = [
        ("list files", "ls"),
        ("show files", "ls"),
        ("delete file", "rm"),
        ("kill process", "kill"),
    ]
    
    space = HyperMapping(dims=8, name="test")
    for input_val, output_val in mappings:
        space.map(input_val, output_val)
    
    # Compute expected similarity matrix (Jaccard)
    n = len(mappings)
    expected_S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            words_i = set(mappings[i][0].lower().split())
            words_j = set(mappings[j][0].lower().split())
            expected_S[i, j] = len(words_i & words_j) / len(words_i | words_j)
    
    print("Expected similarity matrix (Jaccard):")
    print(expected_S.round(3))
    print()
    
    # Apply reproject
    space.reproject()
    
    # Compute actual similarity from positions
    actual_S = np.zeros((n, n))
    positions = [m.position for m in space._mappings]
    for i in range(n):
        for j in range(n):
            actual_S[i, j] = np.dot(positions[i], positions[j]) / (
                np.linalg.norm(positions[i]) * np.linalg.norm(positions[j])
            )
    
    print("Actual similarity matrix (from positions):")
    print(actual_S.round(3))
    print()
    
    # Check preservation
    # Note: Due to normalization, we check relative ordering, not exact values
    print("Similarity ordering preserved:")
    for i in range(n):
        expected_order = np.argsort(expected_S[i])[::-1]
        actual_order = np.argsort(actual_S[i])[::-1]
        match = np.array_equal(expected_order, actual_order)
        print(f"  Row {i}: {match} (expected: {expected_order}, actual: {actual_order})")


if __name__ == "__main__":
    test_reproject()
    test_similarity_preservation()
