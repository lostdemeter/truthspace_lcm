#!/usr/bin/env python3
"""
TruthSpace LCM Demo - Hypergeometric Resolution

Demonstrates how φ-MAX encoding enables semantic resolution
without training, keywords, or neural networks.

Run with: python scripts/demo.py
"""

from truthspace_lcm import TruthSpace, KnowledgeGapError, PHI


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def demo_encoding():
    """Show how φ-MAX encoding works."""
    print_header("φ-MAX ENCODING")
    
    ts = TruthSpace()
    
    print("\nThe golden ratio φ ≈ 1.618 creates natural level separation.")
    print("MAX per dimension prevents synonym over-counting.")
    print()
    
    # Show level separation
    print("Level separation (φ^level):")
    for level in range(6):
        value = PHI ** level
        print(f"  Level {level}: {value:.3f}")
    
    print()
    print("Synonym collapse (Sierpinski property):")
    
    synonyms = ["storage", "disk storage", "disk space storage"]
    for text in synonyms:
        pos = ts._encode(text)
        dim5 = pos[5]
        print(f"  \"{text}\" → dim5 = {dim5:.3f}")
    
    print()
    print("All synonyms encode to the SAME value!")


def demo_resolution():
    """Show resolution in action."""
    print_header("GEOMETRIC RESOLUTION")
    
    ts = TruthSpace()
    
    queries = [
        ("list directory contents", "ls"),
        ("show disk space", "df"),
        ("copy files", "cp"),
        ("move files", "mv"),
        ("find files", "find"),
        ("search text in files", "grep"),
        ("show running processes", "ps"),
        ("kill process", "kill"),
        ("compress files", "tar"),
        ("download from url", "curl"),
    ]
    
    print("\nQuery → Command (similarity)")
    print("-" * 40)
    
    correct = 0
    for query, expected in queries:
        try:
            output, entry, sim = ts.resolve(query)
            status = "✓" if output == expected else "✗"
            if output == expected:
                correct += 1
            print(f"{status} \"{query}\" → {output} ({sim:.2f})")
        except KnowledgeGapError as e:
            print(f"✗ \"{query}\" → no match")
    
    print()
    print(f"Accuracy: {correct}/{len(queries)} ({100*correct/len(queries):.0f}%)")


def demo_explain():
    """Show how to explain a resolution."""
    print_header("EXPLAINABILITY")
    
    ts = TruthSpace()
    
    print("\nHow does 'show disk space' resolve to 'df'?")
    print()
    print(ts.explain("show disk space"))


def demo_add_knowledge():
    """Show how to add new knowledge."""
    print_header("ADDING KNOWLEDGE")
    
    ts = TruthSpace()
    
    print("\nBefore: 'deploy application' has no match")
    try:
        ts.resolve("deploy application")
    except KnowledgeGapError as e:
        print(f"  KnowledgeGapError (best: {e.best_similarity:.2f})")
    
    print("\nAdding knowledge: ts.store('kubectl apply', 'deploy application')")
    ts.store("kubectl apply", "deploy application")
    
    print("\nAfter:")
    output, entry, sim = ts.resolve("deploy application")
    print(f"  'deploy application' → {output} ({sim:.2f})")


def main():
    print("\n" + "=" * 60)
    print(" TRUTHSPACE LCM - HYPERGEOMETRIC RESOLUTION")
    print(" No training. No keywords. Pure geometry.")
    print("=" * 60)
    
    demo_encoding()
    demo_resolution()
    demo_explain()
    demo_add_knowledge()
    
    print_header("KEY INSIGHTS")
    print("""
1. φ-MAX encoding: φ^level with MAX per dimension
2. Sierpinski property: Overlapping activations don't stack
3. φ-weighted distance: Actions > Domains > Relations
4. Pure geometry replaces trained neural networks

This is a showcase of how hypergeometry can replace
LLM/LCM functionality with mathematical resolution.
""")


if __name__ == "__main__":
    main()
