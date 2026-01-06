#!/usr/bin/env python3
"""
Demo: Dynamic Quaternion Layers in Chat Application

Demonstrates the integration of:
1. Dynamic Dimension Registry (Design 105)
2. Quaternion Encoder (Design 104)
3. φ-Zipf Weighting (Design 039)
4. Tachyon Hypothesis Navigation (Design 053)

Run with:
    python -m truthspace_lcm.practical_applications.chat.demo_quaternion

Author: Lesley Gushurst
License: GPLv3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from truthspace_lcm.core import (
    ChatPipeline, ChatConfig,
    QuaternionEncoder, QuaternionPosition,
    DynamicDimensionRegistry,
)


def demo_basic_encoding():
    """Demonstrate basic quaternion encoding."""
    print("=" * 70)
    print("DEMO 1: BASIC QUATERNION ENCODING")
    print("=" * 70)
    print()
    
    config = ChatConfig(use_quaternion=True)
    pipeline = ChatPipeline(config)
    
    test_sentences = [
        "The brave king spoke loudly to his subjects",
        "The cowardly queen whispered softly to her servants",
        "The old man walked slowly down the dark road",
        "The young girl ran quickly through the bright garden",
        "The rich lord lived in a grand palace",
        "The poor peasant dwelt in a humble cottage",
    ]
    
    print("Encoding sentences to quaternion positions:")
    print("-" * 50)
    
    for sentence in test_sentences:
        dims = pipeline.get_text_dimensions(sentence)
        print(f"\n'{sentence}'")
        print(f"  Dimensions: {dims}")


def demo_similarity():
    """Demonstrate quaternion-based similarity."""
    print()
    print("=" * 70)
    print("DEMO 2: QUATERNION SIMILARITY")
    print("=" * 70)
    print()
    
    config = ChatConfig(use_quaternion=True)
    pipeline = ChatPipeline(config)
    
    query = "The brave king shouted commands"
    candidates = [
        "The bold prince spoke loudly",
        "The cowardly queen whispered secrets",
        "The timid princess murmured softly",
        "The old servant walked slowly",
        "The wise sage gave advice",
    ]
    
    print(f"Query: '{query}'")
    print(f"  Dimensions: {pipeline.get_text_dimensions(query)}")
    print()
    print("Candidates (sorted by similarity):")
    print("-" * 50)
    
    results = []
    for candidate in candidates:
        sim = pipeline.quaternion_similarity(query, candidate)
        dims = pipeline.get_text_dimensions(candidate)
        results.append((candidate, sim, dims))
    
    results.sort(key=lambda x: -x[1])
    
    for candidate, sim, dims in results:
        print(f"\n  [{sim:.3f}] '{candidate}'")
        print(f"          {dims}")


def demo_corpus_ingestion():
    """Demonstrate corpus ingestion and entity discovery."""
    print()
    print("=" * 70)
    print("DEMO 3: CORPUS INGESTION & ENTITY DISCOVERY")
    print("=" * 70)
    print()
    
    config = ChatConfig(use_quaternion=True)
    pipeline = ChatPipeline(config)
    
    # Sample corpus (Pride and Prejudice style)
    corpus = """
    Mr Darcy was a proud gentleman of considerable fortune.
    Miss Elizabeth Bennet was a clever and witty young lady.
    Mr Bingley was an amiable and agreeable man.
    Miss Jane Bennet was a beautiful and gentle woman.
    
    Mr Darcy spoke coldly to Elizabeth at the ball.
    Elizabeth replied with spirit and intelligence.
    Jane smiled warmly at Mr Bingley.
    Bingley was delighted by Jane's kindness.
    
    The rich Mr Darcy owned the grand estate of Pemberley.
    The poor Bennet family lived in a modest house.
    Mr Collins was a foolish and pompous clergyman.
    Lady Catherine was a proud and arrogant aristocrat.
    
    Elizabeth walked briskly through the gardens.
    Darcy rode swiftly across his vast lands.
    Jane sat quietly in the drawing room.
    Bingley danced happily at every ball.
    """
    
    print("Ingesting corpus...")
    pipeline.ingest_corpus(corpus)
    
    print("\nDiscovering entities...")
    entities = pipeline.discover_entities()
    
    print("\nTop entities discovered:")
    print("-" * 50)
    for name, score, dim_density in entities[:10]:
        print(f"  {name:15s} score={score:.1f} dim_density={dim_density:.2f}")
    
    print("\nEncoding sentences from corpus:")
    print("-" * 50)
    
    test_sentences = [
        "Mr Darcy spoke proudly",
        "Elizabeth replied cleverly",
        "Jane smiled gently",
        "Bingley danced happily",
    ]
    
    for sentence in test_sentences:
        dims = pipeline.get_text_dimensions(sentence)
        print(f"\n  '{sentence}'")
        print(f"    → {dims}")


def demo_dimension_layers():
    """Demonstrate the quaternion layer structure."""
    print()
    print("=" * 70)
    print("DEMO 4: QUATERNION LAYER STRUCTURE")
    print("=" * 70)
    print()
    
    config = ChatConfig(use_quaternion=True)
    pipeline = ChatPipeline(config)
    
    text = "The brave king spoke loudly and formally"
    
    result = pipeline.encode_quaternion_with_description(text)
    if result:
        pos, desc = result
        
        print(f"Text: '{text}'")
        print()
        print("Quaternion Position Q = w + xi + yj + zk")
        print("-" * 50)
        print(f"w (Semantic):    {pos.w.tolist()}")
        print(f"x (Grammatical): {pos.x.tolist()}")
        print(f"y (Contextual):  {pos.y.tolist()}")
        print(f"z (Dynamic):     [{len(pos.z)} dimensions active]")
        print()
        print("Layer Descriptions:")
        print("-" * 50)
        print(f"  Semantic:    {desc.get('semantic', {})}")
        print(f"  Grammatical: {desc.get('grammatical', {})}")
        print(f"  Contextual:  {desc.get('contextual', {})}")
        print(f"  Dynamic (z): {desc.get('z_active', {})}")


def demo_regality_example():
    """Demonstrate the regality dimension example from Design 105."""
    print()
    print("=" * 70)
    print("DEMO 5: REGALITY DIMENSION (Design 105 Example)")
    print("=" * 70)
    print()
    
    config = ChatConfig(use_quaternion=True)
    pipeline = ChatPipeline(config)
    
    sentences = [
        ("she put out the table ware for guests", "neutral female, common"),
        ("he put out the finery for company", "male, regal"),
        ("they laid out the china for the visitors", "neutral, elevated"),
    ]
    
    print("Same core action, different dimensions:")
    print("-" * 50)
    
    for sentence, description in sentences:
        dims = pipeline.get_text_dimensions(sentence)
        print(f"\n'{sentence}'")
        print(f"  Expected: {description}")
        print(f"  Dimensions: {dims}")
    
    print()
    print("KEY INSIGHT:")
    print("  The core action (setting table for visitors) is identical.")
    print("  The ONLY difference is in the dynamic z-dimensions:")
    print("    - gender: she(-1) vs he(+1) vs they(0)")
    print("    - regality: table ware(0) vs finery(+1.5) vs china(+1)")


def demo_chat_with_dimensions():
    """Demonstrate chat with dimension awareness."""
    print()
    print("=" * 70)
    print("DEMO 6: CHAT WITH DIMENSION AWARENESS")
    print("=" * 70)
    print()
    
    config = ChatConfig(use_quaternion=True)
    pipeline = ChatPipeline(config)
    
    queries = [
        "What is Python?",
        "Tell me about the brave king",
        "How does the cowardly queen rule?",
    ]
    
    print("Chat queries with dimension analysis:")
    print("-" * 50)
    
    for query in queries:
        response = pipeline.chat(query)
        dims = pipeline.get_text_dimensions(query)
        
        print(f"\nQuery: '{query}'")
        print(f"  Dimensions: {dims}")
        print(f"  Response: {response[:100]}...")


def main():
    """Run all demos."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " DYNAMIC QUATERNION LAYERS DEMO ".center(68) + "║")
    print("║" + " Design 104-105: Scalable Dimensional Encoding ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    demo_basic_encoding()
    demo_similarity()
    demo_corpus_ingestion()
    demo_dimension_layers()
    demo_regality_example()
    demo_chat_with_dimensions()
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("The dynamic quaternion layer system provides:")
    print("  1. Structured layers (w, x, y) for semantic/grammatical/contextual")
    print("  2. Dynamic z-layer for emergent dimensions (gender, regality, etc.)")
    print("  3. φ-Zipf weighting for geometric importance")
    print("  4. Tachyon navigation for entity discovery")
    print()
    print("Total dimensions: 12 structured + 15 dynamic = 27 dimensions")
    print("Scalable to 128+ dimensions as needed.")
    print()


if __name__ == "__main__":
    main()
