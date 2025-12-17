#!/usr/bin/env python3
"""
Experiment 1: Domain Centroid Definition

Question: How should we define domain centroids?

Options tested:
1. From signature primitives (encode the primitive keywords)
2. From example queries (mean of encoded examples)
3. Emergent from entries (mean of encoded entry descriptions)

We test each approach on domain classification accuracy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from truthspace_lcm import TruthSpace
from truthspace_lcm.core.truthspace import PHI


@dataclass
class DomainDefinition:
    """Definition of a knowledge domain for testing."""
    name: str
    signature_keywords: List[str]  # Keywords that define this domain
    example_queries: List[str]     # Example queries in this domain
    test_queries: List[str]        # Queries to test classification


# =============================================================================
# DOMAIN DEFINITIONS
# =============================================================================

DOMAINS = [
    DomainDefinition(
        name="BASH",
        signature_keywords=["command", "file", "directory", "process", "system", "terminal", "shell"],
        example_queries=[
            "create a file",
            "list directory contents",
            "show running processes",
            "delete the folder",
            "copy file to backup",
        ],
        test_queries=[
            "make a new directory called test",
            "show me disk space",
            "find all python files",
            "kill the process",
            "move file to another location",
        ],
    ),
    DomainDefinition(
        name="SOCIAL",
        signature_keywords=["feel", "feeling", "think", "want", "hello", "thanks", "sorry", "happy", "sad"],
        example_queries=[
            "hello there",
            "how are you feeling",
            "I'm a bit tired today",
            "thanks for your help",
            "I think that's interesting",
        ],
        test_queries=[
            "hey, what's up",
            "I'm feeling a bit out of touch",
            "that makes me happy",
            "sorry about that",
            "I appreciate your help",
        ],
    ),
    DomainDefinition(
        name="CREATIVE",
        signature_keywords=["write", "poem", "story", "compose", "imagine", "creative", "art", "design"],
        example_queries=[
            "write me a poem",
            "create a short story",
            "imagine a world where",
            "compose a haiku",
            "design a logo concept",
        ],
        test_queries=[
            "write something about autumn",
            "tell me a story",
            "create a poem about love",
            "imagine a fantasy world",
            "describe an artistic scene",
        ],
    ),
    DomainDefinition(
        name="INFORMATIONAL",
        signature_keywords=["what", "how", "why", "explain", "define", "describe", "tell", "information"],
        example_queries=[
            "what is photosynthesis",
            "how does gravity work",
            "why is the sky blue",
            "explain quantum mechanics",
            "define democracy",
        ],
        test_queries=[
            "what causes rain",
            "how do computers work",
            "why do we dream",
            "explain the water cycle",
            "describe how plants grow",
        ],
    ),
]


# =============================================================================
# CENTROID COMPUTATION METHODS
# =============================================================================

def compute_centroid_from_keywords(ts: TruthSpace, domain: DomainDefinition) -> np.ndarray:
    """Method 1: Centroid from signature keywords."""
    keyword_text = " ".join(domain.signature_keywords)
    return ts._encode(keyword_text)


def compute_centroid_from_examples(ts: TruthSpace, domain: DomainDefinition) -> np.ndarray:
    """Method 2: Centroid from mean of example query encodings."""
    encodings = [ts._encode(q) for q in domain.example_queries]
    return np.mean(encodings, axis=0)


def compute_centroid_hybrid(ts: TruthSpace, domain: DomainDefinition) -> np.ndarray:
    """Method 3: Hybrid - combine keywords and examples."""
    keyword_encoding = compute_centroid_from_keywords(ts, domain)
    example_encoding = compute_centroid_from_examples(ts, domain)
    # Weight keywords slightly higher (they're more intentional)
    return 0.6 * keyword_encoding + 0.4 * example_encoding


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify_query(ts: TruthSpace, query: str, centroids: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """Classify a query to the nearest domain centroid."""
    query_pos = ts._encode(query)
    
    best_domain = None
    best_similarity = -np.inf
    
    for domain_name, centroid in centroids.items():
        # Use same similarity metric as TruthSpace
        dist = np.sqrt(np.sum((query_pos - centroid) ** 2))
        similarity = 1.0 / (1.0 + dist)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_domain = domain_name
    
    return best_domain, best_similarity


def evaluate_method(ts: TruthSpace, method_name: str, centroid_fn) -> Dict:
    """Evaluate a centroid computation method."""
    # Compute centroids for all domains
    centroids = {}
    for domain in DOMAINS:
        centroids[domain.name] = centroid_fn(ts, domain)
    
    # Test classification accuracy
    correct = 0
    total = 0
    results = []
    
    for domain in DOMAINS:
        for query in domain.test_queries:
            predicted, confidence = classify_query(ts, query, centroids)
            is_correct = predicted == domain.name
            
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                'query': query,
                'expected': domain.name,
                'predicted': predicted,
                'confidence': confidence,
                'correct': is_correct,
            })
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'method': method_name,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results,
        'centroids': centroids,
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    """Run the centroid definition experiment."""
    print("=" * 70)
    print("EXPERIMENT 1: Domain Centroid Definition")
    print("=" * 70)
    print()
    
    ts = TruthSpace()
    
    methods = [
        ("Keywords Only", compute_centroid_from_keywords),
        ("Examples Only", compute_centroid_from_examples),
        ("Hybrid (60% keywords, 40% examples)", compute_centroid_hybrid),
    ]
    
    all_results = []
    
    for method_name, centroid_fn in methods:
        print(f"\n--- Method: {method_name} ---")
        result = evaluate_method(ts, method_name, centroid_fn)
        all_results.append(result)
        
        print(f"Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
        
        # Show failures
        failures = [r for r in result['results'] if not r['correct']]
        if failures:
            print("Failures:")
            for f in failures:
                print(f"  '{f['query']}' → {f['predicted']} (expected {f['expected']})")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for result in all_results:
        print(f"{result['method']}: {result['accuracy']:.1%}")
    
    best = max(all_results, key=lambda r: r['accuracy'])
    print(f"\nBest method: {best['method']} ({best['accuracy']:.1%})")
    
    # Analyze centroid positions
    print("\n--- Centroid Analysis (Best Method) ---")
    for domain_name, centroid in best['centroids'].items():
        active_dims = [(i, f"{v:.2f}") for i, v in enumerate(centroid) if v > 0.1]
        print(f"{domain_name}: {active_dims[:6]}...")  # Show first 6 active dims
    
    return all_results


if __name__ == "__main__":
    run_experiment()
