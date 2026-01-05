#!/usr/bin/env python3
"""
HyperMapping - Basic Usage Examples

Demonstrates the core functionality of the HyperMapping data structure.

Author: Lesley Gushurst
License: GPLv3
"""

import sys
sys.path.insert(0, '..')

from hypermapping import (
    HyperMapping, 
    TextEncoder, 
    from_pairs,
)


def test_basic_mapping():
    """Test 1: Basic Mapping"""
    print("--- Test 1: Basic Mapping ---")
    space = HyperMapping(dims=8, name="commands")
    
    space.map("list files", "ls")
    space.map("show files", "ls")
    space.map("delete file", "rm")
    space.map("kill process", "kill")
    
    print(f"Created: {space}")
    print(f"Inputs: {space.inputs()}")
    print(f"Outputs: {space.outputs()}")
    print()


def test_forward_query():
    """Test 2: Forward Query (input → output)"""
    print("--- Test 2: Forward Query (input → output) ---")
    space = HyperMapping(dims=8, name="commands")
    space.map("list files", "ls")
    space.map("delete file", "rm")
    
    result = space.forward("list files")
    print(f"  'list files' → {result}")
    
    result = space.forward("delete file")
    print(f"  'delete file' → {result}")
    print()


def test_backward_query():
    """Test 3: Backward Query (output → inputs)"""
    print("--- Test 3: Backward Query (output → inputs) ---")
    space = HyperMapping(dims=8, name="commands")
    space.map("list files", "ls")
    space.map("show files", "ls")
    space.map("delete file", "rm")
    
    results = space.backward("ls", k=5)
    print(f"  'ls' ← ")
    for r in results:
        print(f"    {r}")
    print()


def test_text_encoder():
    """Test 4: Text Encoder with Similarity"""
    print("--- Test 4: Text Encoder with Similarity ---")
    encoder = TextEncoder(dims=8)
    
    # Learn from corpus
    corpus = [
        "list files", "show files", "display files",
        "delete file", "remove file",
        "kill process", "stop process",
    ]
    encoder.learn(corpus)
    encoder.add_synonyms([
        ["list", "show", "display", "enumerate"],
        ["delete", "remove", "erase"],
        ["kill", "stop", "terminate"],
    ])
    
    text_space = HyperMapping(dims=8, encoder=encoder, name="text_commands")
    text_space.map("list files", "ls")
    text_space.map("show files", "ls")
    text_space.map("delete file", "rm")
    text_space.map("kill process", "kill")
    
    print("Query: 'display files'")
    result = text_space.forward("display files")
    print(f"  → {result}")
    
    print("Query: 'remove file'")
    result = text_space.forward("remove file")
    print(f"  → {result}")
    
    print("Query: 'terminate process'")
    result = text_space.forward("terminate process")
    print(f"  → {result}")
    print()
    
    return text_space


def test_pipeline():
    """Test 5: Pipeline"""
    print("--- Test 5: Pipeline ---")
    intent_space = HyperMapping(dims=8, name="intent")
    intent_space.map("file", "file_ops")
    intent_space.map("process", "proc_ops")
    
    cmd_space = HyperMapping(dims=8, name="commands")
    cmd_space.map("file_ops", "ls")
    cmd_space.map("proc_ops", "ps")
    
    pipeline = intent_space | cmd_space
    print(f"Pipeline: {pipeline}")
    
    result = pipeline("file")
    print(f"  'file' → {result}")
    
    result = pipeline("process")
    print(f"  'process' → {result}")
    print()


def test_from_pairs():
    """Test 6: from_pairs() convenience function"""
    print("--- Test 6: from_pairs() ---")
    quick_space = from_pairs([
        ("hello", "world"),
        ("foo", "bar"),
        ("python", "programming"),
    ], name="quick")
    print(f"Created: {quick_space}")
    print()


def test_serialization(space: HyperMapping):
    """Test 7: Serialization"""
    print("--- Test 7: Serialization ---")
    space.save("/tmp/hypermapping_test.json")
    print("Saved to /tmp/hypermapping_test.json")
    
    loaded = HyperMapping.load("/tmp/hypermapping_test.json")
    print(f"Loaded: {loaded}")
    result = loaded.forward("list files")
    print(f"  'list files' → {result}")
    print()


def test_iteration(space: HyperMapping):
    """Test 8: Iteration"""
    print("--- Test 8: Iteration ---")
    print("All mappings:")
    for mapping in space:
        print(f"  {mapping}")
    print()


def main():
    print("=" * 60)
    print("  HYPERMAPPING - Bidirectional Hyperdimensional Mapping")
    print("=" * 60)
    print()
    
    test_basic_mapping()
    test_forward_query()
    test_backward_query()
    text_space = test_text_encoder()
    test_pipeline()
    test_from_pairs()
    test_serialization(text_space)
    test_iteration(text_space)
    
    print("=" * 60)
    print("  EXAMPLES COMPLETE")
    print("=" * 60)
    print()
    print("HyperMapping provides:")
    print("  ✓ Bidirectional mappings (input ↔ output)")
    print("  ✓ Forward query (input → output)")
    print("  ✓ Backward query (output → inputs)")
    print("  ✓ Similarity-based matching")
    print("  ✓ Pluggable encoders")
    print("  ✓ Learning (feedback, attract, repel)")
    print("  ✓ Chaining with | operator")
    print("  ✓ Serialization")


if __name__ == "__main__":
    main()
