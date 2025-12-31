"""
Comparison Test: Old vs New Deficiency Detection

Compares:
- OLD: Pattern-matching based (DeficiencyDetectorGear)
- NEW: Shape-based (FoldingDeficiencyDetector)

Author: Lesley Gushurst
License: GPLv3
"""

import sys
from typing import List, Dict
from .folding_deficiency import (
    FoldingDeficiencyDetector, FoldingStructure, 
    ShapeDeficiency, ShapeDeficiencyType,
    compare_detection_methods
)
from .gear_improvement_loop import DeficiencyDetectorGear, TestCase, DeficiencyType


def run_comparison_test():
    """
    Run comprehensive comparison between old and new methods.
    """
    print("=" * 70)
    print("DEFICIENCY DETECTION COMPARISON: Old vs New")
    print("=" * 70)
    
    # Test cases: (expected, actual, expected_contains, description)
    test_cases = [
        # Good matches (should detect NO deficiency)
        (
            "Captain Ahab commanded the ship. The captain led the crew. Ahab hunted the whale.",
            "Admiral Kirk commanded the starship. The admiral led the crew. Kirk explored the galaxy.",
            ["captain", "ship"],
            "Same structure, different domain"
        ),
        (
            "Captain Ahab commanded the ship. The captain led the crew. Ahab hunted the whale.",
            "Captian Ahab comanded the shipp. The captian led the crue. Ahab hunted the wale.",
            ["captain", "ahab"],
            "Same structure with typos"
        ),
        
        # Deficient outputs (should detect deficiency)
        (
            "Captain Ahab commanded the ship. The captain led the crew. Ahab hunted the whale.",
            "Ahab hunted.",
            ["captain", "ship", "crew"],
            "Too short / incomplete"
        ),
        (
            "Captain Ahab commanded the ship. The captain led the crew. Ahab hunted the whale.",
            "The whale. The ship. The captain. The crew. The ocean.",
            ["commanded", "led", "hunted"],
            "Wrong structure (list vs narrative)"
        ),
        (
            "Captain Ahab commanded the ship. The captain led the crew. Ahab hunted the whale.",
            "The weather was nice. The sun was shining. Birds were singing.",
            ["captain", "ahab", "whale"],
            "Irrelevant content"
        ),
        (
            "Captain Ahab commanded the ship. The captain led the crew. Ahab hunted the whale.",
            "Something happened. Things occurred. Events transpired.",
            ["captain", "ahab", "whale"],
            "Vague content"
        ),
        (
            "Captain Ahab commanded the ship. The captain led the crew. Ahab hunted the whale.",
            "Ahab commanded the ship. He led the crew. He hunted the whale.",
            ["captain"],
            "Missing self-references (uses pronouns)"
        ),
    ]
    
    results = []
    
    for expected, actual, expected_contains, description in test_cases:
        print(f"\n{'='*70}")
        print(f"TEST: {description}")
        print(f"{'='*70}")
        print(f"Expected: {expected[:50]}...")
        print(f"Actual: {actual[:50]}...")
        
        comparison = compare_detection_methods(expected, actual, expected_contains)
        
        print(f"\n--- OLD METHOD (Pattern Matching) ---")
        print(f"   Deficiencies found: {comparison['old_method']['deficiency_count']}")
        print(f"   Max severity: {comparison['old_method']['max_severity']:.2f}")
        for d in comparison['old_method']['deficiencies'][:3]:
            print(f"      - {d['type']}: {d['description'][:50]}")
        
        print(f"\n--- NEW METHOD (Shape-Based) ---")
        print(f"   Type: {comparison['new_method']['type']}")
        print(f"   Severity: {comparison['new_method']['severity']:.2f}")
        print(f"   Shape similarity: {comparison['new_method']['shape_similarity']:.3f}")
        print(f"   Fold ratio: {comparison['new_method']['fold_ratio']:.2f}")
        print(f"   Description: {comparison['new_method']['description']}")
        if comparison['new_method']['missing_fold_words']:
            print(f"   Missing fold words: {comparison['new_method']['missing_fold_words']}")
        
        results.append({
            'description': description,
            'old_severity': comparison['old_method']['max_severity'],
            'new_severity': comparison['new_method']['severity'],
            'new_type': comparison['new_method']['type'],
            'shape_sim': comparison['new_method']['shape_similarity'],
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\n| Test Case | Old Severity | New Severity | New Type | Shape Sim |")
    print("|-----------|--------------|--------------|----------|-----------|")
    for r in results:
        print(f"| {r['description'][:25]:25} | {r['old_severity']:.2f} | {r['new_severity']:.2f} | {r['new_type'][:10]:10} | {r['shape_sim']:.3f} |")
    
    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)
    
    # Good matches should have low severity
    good_matches = [r for r in results if 'Same structure' in r['description'] or 'typos' in r['description']]
    bad_matches = [r for r in results if r not in good_matches]
    
    if good_matches:
        good_old_avg = sum(r['old_severity'] for r in good_matches) / len(good_matches)
        good_new_avg = sum(r['new_severity'] for r in good_matches) / len(good_matches)
        print(f"\nGood matches (should be LOW severity):")
        print(f"   Old method avg severity: {good_old_avg:.2f}")
        print(f"   New method avg severity: {good_new_avg:.2f}")
        print(f"   Winner: {'NEW' if good_new_avg < good_old_avg else 'OLD' if good_old_avg < good_new_avg else 'TIE'}")
    
    if bad_matches:
        bad_old_avg = sum(r['old_severity'] for r in bad_matches) / len(bad_matches)
        bad_new_avg = sum(r['new_severity'] for r in bad_matches) / len(bad_matches)
        print(f"\nBad matches (should be HIGH severity):")
        print(f"   Old method avg severity: {bad_old_avg:.2f}")
        print(f"   New method avg severity: {bad_new_avg:.2f}")
        print(f"   Winner: {'NEW' if bad_new_avg > bad_old_avg else 'OLD' if bad_old_avg > bad_new_avg else 'TIE'}")
    
    # Discrimination
    if good_matches and bad_matches:
        old_separation = bad_old_avg - good_old_avg
        new_separation = bad_new_avg - good_new_avg
        print(f"\nDiscrimination (bad - good severity):")
        print(f"   Old method separation: {old_separation:.2f}")
        print(f"   New method separation: {new_separation:.2f}")
        print(f"   Better discrimination: {'NEW' if new_separation > old_separation else 'OLD'}")
    
    # Error tolerance
    typo_test = [r for r in results if 'typos' in r['description']]
    if typo_test:
        print(f"\nError tolerance (typos test):")
        print(f"   Old method severity: {typo_test[0]['old_severity']:.2f}")
        print(f"   New method severity: {typo_test[0]['new_severity']:.2f}")
        print(f"   Shape similarity: {typo_test[0]['shape_sim']:.3f}")
        print(f"   Better error tolerance: {'NEW' if typo_test[0]['new_severity'] < typo_test[0]['old_severity'] else 'OLD'}")
    
    return results


if __name__ == "__main__":
    results = run_comparison_test()
