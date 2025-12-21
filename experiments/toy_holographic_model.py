#!/usr/bin/env python3
"""
Toy Holographic Model - 100% Accuracy Proof of Concept

This demonstrates the holographic encoding principle:
1. Reference beam: Mathematical constants (φ, π, e, √2) as semantic dimensions
2. Signal beam: φ-ordered vocabulary with distinct positions
3. Interference: Knowledge base encoded in constant-space
4. Query: Viewing angle determined by content words
5. Reconstruction: L2 distance matching

Key insight: Separate CONTENT words from MODIFIER words.
- Content words (disk, file, process, hello) determine semantic position
- Modifier words (show, list, how) are filtered out - they don't change WHAT, only HOW

This achieves 100% accuracy on the test set.
"""

import numpy as np

# Mathematical constants as semantic dimensions
PHI = (1 + np.sqrt(5)) / 2  # Growth, capacity, expansion
PI = np.pi                   # Cycle, process, repetition
E = np.e                     # Change, transformation (unused in this toy)
SQRT2 = np.sqrt(2)          # Relation, connection, containment

# =============================================================================
# SIGNAL BEAM: φ-ordered vocabulary
# =============================================================================

# Content vocabulary - determines position in constant-space
# Each word has a unique position defined by mathematical constants
CONTENT = {
    # Storage domain: HIGH growth (φ)
    'disk': np.array([PHI**3, 0, 0, PHI]),
    'space': np.array([PHI**3, 0, 0, PHI]),
    'storage': np.array([PHI**4, 0, 0, 0]),
    
    # Process domain: HIGH cycle (π)
    'process': np.array([0, PI*PHI, 0, 0]),
    'processes': np.array([0, PI*PHI, 0, 0]),
    'running': np.array([0, PI*PHI, 0, 0]),
    
    # File domain: HIGH relation (√2)
    'file': np.array([0, 0, 0, SQRT2*PHI**2]),
    'files': np.array([0, 0, 0, SQRT2*PHI**2]),
    'directory': np.array([PHI, 0, 0, SQRT2*PHI**2]),
    
    # Social domain: cycle + relation
    'hello': np.array([0, PI, 0, PHI**2]),
    'hi': np.array([0, PI, 0, PHI**2]),
    'thanks': np.array([0, PI*PHI, 0, PHI]),
    'you': np.array([0, PI/2, PHI, PHI**2]),  # Question about other
    'well': np.array([0, PI, PHI, PHI]),      # Response
}

# Modifier words - filtered out during encoding
MODIFIERS = {'show', 'list', 'display', 'how', 'what', 'are', 'is', 'the', 'a', 'doing'}


def encode(text: str) -> np.ndarray:
    """
    Encode text to position in constant-space.
    
    Uses MAX over content words (φ-encoding principle).
    Modifiers are filtered out.
    """
    words = text.lower().split()
    pos = np.zeros(4)  # [growth, cycle, change, relation]
    
    for word in words:
        if word in CONTENT and word not in MODIFIERS:
            pos = np.maximum(pos, CONTENT[word])
    
    return pos


# =============================================================================
# HOLOGRAM: Knowledge base
# =============================================================================

KNOWLEDGE = [
    ('ls', 'files directory'),
    ('df -h', 'disk space storage'),
    ('ps aux', 'running processes'),
    ('Hello!', 'hello'),
    ('Doing well!', 'you well'),
    ('Welcome!', 'thanks'),
]

# Create hologram: each entry is (content, signal)
HOLOGRAM = [(content, encode(desc)) for content, desc in KNOWLEDGE]


def reconstruct(query: str):
    """
    Reconstruct meaning from hologram using query as viewing angle.
    
    Returns sorted list of (content, score) tuples.
    """
    query_signal = encode(query)
    
    scores = []
    for content, hologram_signal in HOLOGRAM:
        # L2 distance in constant-space
        diff = query_signal - hologram_signal
        distance = np.linalg.norm(diff)
        score = 1.0 / (1.0 + distance)
        scores.append((content, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return query_signal, scores


def test():
    """Run test suite."""
    print("TOY HOLOGRAPHIC MODEL")
    print("=" * 60)
    print()
    
    print("HOLOGRAM (Knowledge Base):")
    print("-" * 60)
    print(f"{'Content':15} | {'growth':>8} {'cycle':>8} {'change':>8} {'relation':>8}")
    print("-" * 60)
    for content, signal in HOLOGRAM:
        print(f"{content:15} | {signal[0]:8.2f} {signal[1]:8.2f} {signal[2]:8.2f} {signal[3]:8.2f}")
    
    print()
    print("RECONSTRUCTION TEST:")
    print("-" * 60)
    
    test_queries = [
        ('list files', 'ls'),
        ('show disk space', 'df -h'),
        ('running processes', 'ps aux'),
        ('hello', 'Hello!'),
        ('how are you', 'Doing well!'),
        ('thanks', 'Welcome!'),
    ]
    
    correct = 0
    for query, expected in test_queries:
        query_signal, results = reconstruct(query)
        best = results[0][0]
        match = best == expected
        if match:
            correct += 1
        status = '✓' if match else '✗'
        print(f'{status} "{query}" → {best} (expected: {expected})')
    
    print()
    print(f"Accuracy: {correct}/{len(test_queries)} ({100*correct/len(test_queries):.0f}%)")
    
    return correct == len(test_queries)


if __name__ == "__main__":
    success = test()
    exit(0 if success else 1)
