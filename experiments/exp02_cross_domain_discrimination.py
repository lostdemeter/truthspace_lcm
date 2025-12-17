#!/usr/bin/env python3
"""
Experiment 2: Cross-Domain Discrimination

Question: Can we discriminate overlapping concepts across domains?

The challenge: Words like "create", "touch", "run" appear in multiple domains
with different meanings:
- "create a file" → BASH
- "create a poem" → CREATIVE
- "touch the file" → BASH
- "feeling out of touch" → SOCIAL
- "run the command" → BASH
- "run in the park" → PHYSICAL/SOCIAL

We test whether the surrounding context provides enough discrimination.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from truthspace_lcm import TruthSpace


@dataclass
class OverlapTestCase:
    """A test case with overlapping keyword but different domain."""
    query: str
    expected_domain: str
    overlapping_word: str


# =============================================================================
# TEST CASES - Queries with overlapping keywords
# =============================================================================

OVERLAP_TESTS = [
    # "create" overlap
    OverlapTestCase("create a file called test.txt", "BASH", "create"),
    OverlapTestCase("create a poem about nature", "CREATIVE", "create"),
    OverlapTestCase("create a new directory", "BASH", "create"),
    OverlapTestCase("create an imaginative story", "CREATIVE", "create"),
    
    # "touch" overlap
    OverlapTestCase("touch the file to update timestamp", "BASH", "touch"),
    OverlapTestCase("I'm feeling out of touch lately", "SOCIAL", "touch"),
    OverlapTestCase("touch test.txt", "BASH", "touch"),
    OverlapTestCase("that story really touched me", "SOCIAL", "touch"),
    
    # "run" overlap
    OverlapTestCase("run the script", "BASH", "run"),
    OverlapTestCase("I went for a run this morning", "SOCIAL", "run"),
    OverlapTestCase("run this command", "BASH", "run"),
    OverlapTestCase("my mind is running wild", "SOCIAL", "run"),
    
    # "show" overlap
    OverlapTestCase("show me the files", "BASH", "show"),
    OverlapTestCase("show me how you feel", "SOCIAL", "show"),
    OverlapTestCase("show disk space", "BASH", "show"),
    OverlapTestCase("I want to show my appreciation", "SOCIAL", "show"),
    
    # "make" overlap
    OverlapTestCase("make a directory", "BASH", "make"),
    OverlapTestCase("make me happy", "SOCIAL", "make"),
    OverlapTestCase("make the folder", "BASH", "make"),
    OverlapTestCase("that makes sense to me", "SOCIAL", "make"),
    
    # "find" overlap
    OverlapTestCase("find all log files", "BASH", "find"),
    OverlapTestCase("I need to find myself", "SOCIAL", "find"),
    OverlapTestCase("find the config file", "BASH", "find"),
    OverlapTestCase("help me find meaning", "SOCIAL", "find"),
    
    # "write" overlap
    OverlapTestCase("write to the file", "BASH", "write"),
    OverlapTestCase("write a beautiful poem", "CREATIVE", "write"),
    OverlapTestCase("write data to disk", "BASH", "write"),
    OverlapTestCase("write me a story", "CREATIVE", "write"),
    
    # "delete" / "remove" overlap
    OverlapTestCase("delete the file", "BASH", "delete"),
    OverlapTestCase("I want to delete these negative thoughts", "SOCIAL", "delete"),
    OverlapTestCase("remove the directory", "BASH", "remove"),
    OverlapTestCase("remove my doubts", "SOCIAL", "remove"),
]


# =============================================================================
# DOMAIN DEFINITIONS (same as exp01)
# =============================================================================

DOMAIN_KEYWORDS = {
    "BASH": ["command", "file", "directory", "process", "system", "terminal", "shell", "disk", "folder"],
    "SOCIAL": ["feel", "feeling", "think", "want", "hello", "thanks", "sorry", "happy", "sad", "myself", "me", "I"],
    "CREATIVE": ["poem", "story", "compose", "imagine", "creative", "art", "design", "beautiful", "imaginative"],
    "INFORMATIONAL": ["what", "how", "why", "explain", "define", "describe", "tell", "information"],
}


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_domain_centroids(ts: TruthSpace) -> dict:
    """Compute centroids for each domain using keywords."""
    centroids = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        keyword_text = " ".join(keywords)
        centroids[domain] = ts._encode(keyword_text)
    return centroids


def classify_query(ts: TruthSpace, query: str, centroids: dict) -> Tuple[str, dict]:
    """Classify query and return all domain scores."""
    query_pos = ts._encode(query)
    
    scores = {}
    for domain_name, centroid in centroids.items():
        dist = np.sqrt(np.sum((query_pos - centroid) ** 2))
        scores[domain_name] = 1.0 / (1.0 + dist)
    
    best_domain = max(scores, key=scores.get)
    return best_domain, scores


def analyze_encoding(ts: TruthSpace, query: str) -> dict:
    """Analyze which dimensions are activated by a query."""
    pos = ts._encode(query)
    active = {}
    for i, v in enumerate(pos):
        if v > 0.1:
            active[i] = v
    return active


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    """Run the cross-domain discrimination experiment."""
    print("=" * 70)
    print("EXPERIMENT 2: Cross-Domain Discrimination")
    print("=" * 70)
    print()
    
    ts = TruthSpace()
    centroids = compute_domain_centroids(ts)
    
    # Group tests by overlapping word
    by_overlap = {}
    for test in OVERLAP_TESTS:
        if test.overlapping_word not in by_overlap:
            by_overlap[test.overlapping_word] = []
        by_overlap[test.overlapping_word].append(test)
    
    total_correct = 0
    total_tests = 0
    
    for overlap_word, tests in by_overlap.items():
        print(f"\n--- Overlapping word: '{overlap_word}' ---")
        
        correct = 0
        for test in tests:
            predicted, scores = classify_query(ts, test.query, centroids)
            is_correct = predicted == test.expected_domain
            
            if is_correct:
                correct += 1
                total_correct += 1
            total_tests += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {status} \"{test.query[:40]}...\"")
            print(f"      Expected: {test.expected_domain}, Got: {predicted}")
            print(f"      Scores: {', '.join(f'{k}={v:.2f}' for k, v in sorted(scores.items(), key=lambda x: -x[1]))}")
        
        print(f"  Accuracy for '{overlap_word}': {correct}/{len(tests)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Overall accuracy: {total_correct}/{total_tests} ({100*total_correct/total_tests:.1f}%)")
    
    # Analyze why failures happen
    print("\n--- Failure Analysis ---")
    for test in OVERLAP_TESTS:
        predicted, scores = classify_query(ts, test.query, centroids)
        if predicted != test.expected_domain:
            print(f"\nFailed: \"{test.query}\"")
            print(f"  Expected: {test.expected_domain}, Got: {predicted}")
            
            # Show encoding
            encoding = analyze_encoding(ts, test.query)
            print(f"  Active dimensions: {encoding}")
            
            # Show what the overlapping word alone encodes to
            word_encoding = analyze_encoding(ts, test.overlapping_word)
            print(f"  '{test.overlapping_word}' alone: {word_encoding}")
    
    return total_correct, total_tests


if __name__ == "__main__":
    run_experiment()
