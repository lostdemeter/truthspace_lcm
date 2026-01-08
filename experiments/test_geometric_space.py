#!/usr/bin/env python3
"""
Test the generalized GeometricSpace with multiple domains.

Demonstrates that the same geometric transformation mechanism works for:
1. Text/phrases (like ConceptTransformer)
2. Colors (RGB transformations)
3. Music (note/chord transformations)
4. Code (syntax transformations)

Author: Lesley Gushurst
License: GPLv3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.geometric_space import (
    GeometricSpace, create_text_space, PHI
)


def test_text_transformations():
    """Test with text phrases (like the original ConceptTransformer)."""
    print("=" * 60)
    print("TEST 1: Text Transformations (Tense)")
    print("=" * 60)
    
    space = create_text_space()
    
    # Add tense dimension with explicit levels
    space.add_dimension('tense', {'past': 0, 'present': 1, 'future': 2})
    
    # Learn transformation pairs
    pairs = [
        ("went", "will go", "past", "future"),
        ("sat", "will sit", "past", "future"),
        ("walked", "will walk", "past", "future"),
        ("ran", "will run", "past", "future"),
        ("came", "will come", "past", "future"),
    ]
    
    for src, tgt, src_val, tgt_val in pairs:
        space.learn_pair(src, tgt, 'tense', src_val, tgt_val)
    
    # Compute deltas
    space.compute_deltas()
    
    print(f"\nStats: {space.stats()}")
    
    # Test transformations
    print("\nTransformations:")
    for src, expected, _, _ in pairs:
        result = space.transform(src, 'tense', 'past', 'future')
        status = "✓" if result.success and result.target_item == expected else "✗"
        print(f"  {status} '{src}' → '{result.target_item}' (expected: '{expected}')")
    
    return space


def test_color_transformations():
    """Test with color transformations (brightness, saturation)."""
    print("\n" + "=" * 60)
    print("TEST 2: Color Transformations (Brightness)")
    print("=" * 60)
    
    space = GeometricSpace(
        item_to_key=lambda x: x.lower(),
        key_to_item=lambda x: x
    )
    
    # Brightness dimension
    space.add_dimension('brightness', {'dark': 0, 'medium': 1, 'light': 2})
    
    # Learn color brightness pairs
    pairs = [
        ("navy", "blue", "dark", "medium"),
        ("blue", "sky blue", "medium", "light"),
        ("maroon", "red", "dark", "medium"),
        ("red", "pink", "medium", "light"),
        ("forest green", "green", "dark", "medium"),
        ("green", "lime", "medium", "light"),
        ("charcoal", "gray", "dark", "medium"),
        ("gray", "silver", "medium", "light"),
    ]
    
    for src, tgt, src_val, tgt_val in pairs:
        space.learn_pair(src, tgt, 'brightness', src_val, tgt_val)
    
    space.compute_deltas()
    
    print(f"\nStats: {space.stats()}")
    
    # Test transformations
    print("\nTransformations (dark → medium):")
    test_cases = [("navy", "blue"), ("maroon", "red"), ("forest green", "green")]
    for src, expected in test_cases:
        result = space.transform(src, 'brightness', 'dark', 'medium')
        status = "✓" if result.success and result.target_item == expected else "✗"
        print(f"  {status} '{src}' → '{result.target_item}' (expected: '{expected}')")
    
    print("\nTransformations (medium → light):")
    test_cases = [("blue", "sky blue"), ("red", "pink"), ("green", "lime")]
    for src, expected in test_cases:
        result = space.transform(src, 'brightness', 'medium', 'light')
        status = "✓" if result.success and result.target_item == expected else "✗"
        print(f"  {status} '{src}' → '{result.target_item}' (expected: '{expected}')")
    
    return space


def test_music_transformations():
    """Test with music note/chord transformations."""
    print("\n" + "=" * 60)
    print("TEST 3: Music Transformations (Mode)")
    print("=" * 60)
    
    space = GeometricSpace(
        item_to_key=lambda x: x,
        key_to_item=lambda x: x
    )
    
    # Mode dimension (major/minor)
    space.add_dimension('mode', {'minor': 0, 'major': 1})
    
    # Learn chord mode pairs
    pairs = [
        ("Am", "A", "minor", "major"),
        ("Bm", "B", "minor", "major"),
        ("Cm", "C", "minor", "major"),
        ("Dm", "D", "minor", "major"),
        ("Em", "E", "minor", "major"),
        ("Fm", "F", "minor", "major"),
        ("Gm", "G", "minor", "major"),
    ]
    
    for src, tgt, src_val, tgt_val in pairs:
        space.learn_pair(src, tgt, 'mode', src_val, tgt_val)
    
    space.compute_deltas()
    
    print(f"\nStats: {space.stats()}")
    
    # Test transformations
    print("\nTransformations (minor → major):")
    for src, expected, _, _ in pairs:
        result = space.transform(src, 'mode', 'minor', 'major')
        status = "✓" if result.success and result.target_item == expected else "✗"
        print(f"  {status} '{src}' → '{result.target_item}' (expected: '{expected}')")
    
    return space


def test_code_transformations():
    """Test with code syntax transformations."""
    print("\n" + "=" * 60)
    print("TEST 4: Code Transformations (Language)")
    print("=" * 60)
    
    space = GeometricSpace(
        item_to_key=lambda x: x.strip(),
        key_to_item=lambda x: x
    )
    
    # Language dimension
    space.add_dimension('language', {'python': 0, 'javascript': 1, 'rust': 2})
    
    # Learn syntax pairs
    pairs = [
        # Python → JavaScript
        ("print('hello')", "console.log('hello')", "python", "javascript"),
        ("len(arr)", "arr.length", "python", "javascript"),
        ("True", "true", "python", "javascript"),
        ("False", "false", "python", "javascript"),
        ("None", "null", "python", "javascript"),
        ("def foo():", "function foo() {", "python", "javascript"),
        ("elif", "else if", "python", "javascript"),
        ("and", "&&", "python", "javascript"),
        ("or", "||", "python", "javascript"),
    ]
    
    for src, tgt, src_val, tgt_val in pairs:
        space.learn_pair(src, tgt, 'language', src_val, tgt_val)
    
    space.compute_deltas()
    
    print(f"\nStats: {space.stats()}")
    
    # Test transformations
    print("\nTransformations (Python → JavaScript):")
    for src, expected, _, _ in pairs:
        result = space.transform(src, 'language', 'python', 'javascript')
        status = "✓" if result.success and result.target_item == expected else "✗"
        print(f"  {status} '{src}' → '{result.target_item}'")
    
    return space


def test_dimension_discovery():
    """Test automatic dimension discovery from data."""
    print("\n" + "=" * 60)
    print("TEST 5: Dimension Discovery (Unnamed Dimensions)")
    print("=" * 60)
    
    space = create_text_space()
    
    # Don't predefine dimensions - let them be discovered
    # Learn pairs with arbitrary dimension names
    
    # Some unnamed dimension we'll call "formality_level"
    pairs = [
        ("hi", "hello", "casual", "formal"),
        ("yeah", "yes", "casual", "formal"),
        ("nope", "no", "casual", "formal"),
        ("gonna", "going to", "casual", "formal"),
        ("wanna", "want to", "casual", "formal"),
        ("gotta", "have to", "casual", "formal"),
        ("dunno", "don't know", "casual", "formal"),
    ]
    
    for src, tgt, src_val, tgt_val in pairs:
        space.learn_pair(src, tgt, 'formality', src_val, tgt_val)
    
    space.compute_deltas()
    
    print(f"\nStats: {space.stats()}")
    print(f"Discovered dimension 'formality': {space._dimensions['formality'].levels}")
    
    # Test transformations
    print("\nTransformations (casual → formal):")
    for src, expected, _, _ in pairs:
        result = space.transform(src, 'formality', 'casual', 'formal')
        status = "✓" if result.success and result.target_item == expected else "✗"
        print(f"  {status} '{src}' → '{result.target_item}' (expected: '{expected}')")
    
    return space


def test_temporary_injection():
    """Test temporary injection for unknown items."""
    print("\n" + "=" * 60)
    print("TEST 6: Temporary Injection (Learning New Items)")
    print("=" * 60)
    
    space = create_text_space()
    space.add_dimension('tense', {'past': 0, 'present': 1, 'future': 2})
    
    # Learn some pairs
    pairs = [
        ("went", "will go", "past", "future"),
        ("sat", "will sit", "past", "future"),
    ]
    
    for src, tgt, src_val, tgt_val in pairs:
        space.learn_pair(src, tgt, 'tense', src_val, tgt_val)
    
    space.compute_deltas()
    
    print(f"\nInitial stats: {space.stats()}")
    
    # Try to transform unknown item without injection
    result = space.transform("jumped", 'tense', 'past', 'future', allow_injection=False)
    print(f"\nWithout injection: success={result.success}, reason='{result.failure_reason}'")
    
    # Try with injection
    result = space.transform("jumped", 'tense', 'past', 'future', allow_injection=True)
    print(f"With injection: success={result.success}, was_injected={result.was_injected}")
    print(f"  Result: '{result.target_item}' (nearest neighbor)")
    
    # Simulate LLM providing correct answer
    print("\nSimulating LLM success: 'jumped' → 'will jump'")
    space.promote_temporary("jumped", "will jump", 'tense', 'past', 'future')
    
    print(f"After promotion: {space.stats()}")
    
    # Now transform should work correctly
    result = space.transform("jumped", 'tense', 'past', 'future')
    print(f"\nAfter learning: '{result.source_item}' → '{result.target_item}'")
    print(f"  Success: {result.success}, Confidence: {result.confidence:.2f}")
    
    return space


def test_serialization():
    """Test saving and loading a space."""
    print("\n" + "=" * 60)
    print("TEST 7: Serialization (Save/Load)")
    print("=" * 60)
    
    import tempfile
    import os
    
    # Create a space
    space = create_text_space()
    space.add_dimension('tense', {'past': 0, 'present': 1, 'future': 2})
    
    pairs = [
        ("went", "will go", "past", "future"),
        ("sat", "will sit", "past", "future"),
    ]
    
    for src, tgt, src_val, tgt_val in pairs:
        space.learn_pair(src, tgt, 'tense', src_val, tgt_val)
    
    space.compute_deltas()
    
    print(f"Original stats: {space.stats()}")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    
    space.save(temp_path)
    print(f"Saved to: {temp_path}")
    
    # Load from file
    loaded = GeometricSpace.load(
        temp_path,
        item_to_key=lambda x: x.lower().strip(),
        key_to_item=lambda x: x
    )
    
    print(f"Loaded stats: {loaded.stats()}")
    
    # Test that transformation still works
    result = loaded.transform("went", 'tense', 'past', 'future')
    print(f"\nTransformation after load: 'went' → '{result.target_item}'")
    print(f"  Success: {result.success}")
    
    # Cleanup
    os.unlink(temp_path)
    
    return loaded


def test_multi_dimension():
    """Test items with multiple dimensions."""
    print("\n" + "=" * 60)
    print("TEST 8: Multi-Dimensional Space")
    print("=" * 60)
    
    space = create_text_space()
    
    # Add multiple dimensions
    space.add_dimension('tense', {'past': 0, 'present': 1, 'future': 2})
    space.add_dimension('formality', {'casual': 0, 'formal': 1})
    
    # Learn tense pairs
    space.learn_pair("went", "will go", 'tense', 'past', 'future')
    space.learn_pair("sat", "will sit", 'tense', 'past', 'future')
    
    # Learn formality pairs
    space.learn_pair("went", "proceeded", 'formality', 'casual', 'formal')
    space.learn_pair("sat", "was seated", 'formality', 'casual', 'formal')
    
    space.compute_deltas()
    
    print(f"\nStats: {space.stats()}")
    
    # Test tense transformation
    result = space.transform("went", 'tense', 'past', 'future')
    print(f"\nTense: 'went' → '{result.target_item}'")
    
    # Test formality transformation
    result = space.transform("went", 'formality', 'casual', 'formal')
    print(f"Formality: 'went' → '{result.target_item}'")
    
    return space


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GEOMETRIC SPACE - DOMAIN-AGNOSTIC TRANSFORMATION TESTS")
    print("=" * 60)
    print(f"\nUsing φ = {PHI:.6f}")
    
    test_text_transformations()
    test_color_transformations()
    test_music_transformations()
    test_code_transformations()
    test_dimension_discovery()
    test_temporary_injection()
    test_serialization()
    test_multi_dimension()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
    print("""
Key Insight: The same geometric mechanism works for ANY domain:
- Text phrases (tense, formality)
- Colors (brightness, saturation)
- Music (mode, key)
- Code (language syntax)
- And any other domain with transformation pairs!

The geometry IS the knowledge. No domain-specific logic required.
""")


if __name__ == "__main__":
    main()
