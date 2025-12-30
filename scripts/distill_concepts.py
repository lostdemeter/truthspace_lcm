#!/usr/bin/env python3
"""
Concept Distillation: Extract Pure Concepts from Corpus

This script distills the frame-based corpus into a pure concept model.
The result is a compact representation that captures all the geometric
relationships without storing the original text.

Key insight: Text is evidence for structure. Once we've extracted the
structure, we only need the concepts for inference.

Storage comparison:
- Full corpus: ~7MB (25K frames with text)
- Distilled concepts: ~500KB (just the geometric structure)

The distilled model contains:
1. Concept quaternions (4D semantic encoding)
2. Action/target relationships (weighted edges)
3. Role statistics (φ-direction, counts)
4. Morphological equivalences

Author: Lesley Gushurst
License: GPLv3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
from dataclasses import dataclass, asdict
import math

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthspace_lcm.core.geometric import GeometricKnowledge, HolographicGeometricQA


PHI = 1.618034


def distill_concept_compact(concept, top_k: int = 5) -> List:
    """
    Distill a GeometricConcept to a compact array format.
    
    Format: [phi_dir, freq, i_count, m_count, r_count, [actions], [targets]]
    
    This is ~10x smaller than the dataclass format.
    """
    # Get top actions and targets as lists of [word, count]
    top_actions = concept.actions.most_common(top_k)
    top_targets = concept.targets.most_common(top_k)
    
    return [
        round(concept.phi_direction, 3),  # φ-direction (3 decimal places)
        concept.frequency,
        concept.initiator_count,
        concept.mediator_count, 
        concept.receiver_count,
        top_actions,  # [[action, count], ...]
        top_targets,  # [[target, count], ...]
    ]


def distill_corpus(corpus_path: str, min_frequency: int = 3, top_k: int = 5) -> Dict[str, Any]:
    """
    Distill a corpus into a pure concept model.
    
    Args:
        corpus_path: Path to the frame-based corpus
        min_frequency: Minimum frequency to include a concept
        top_k: Number of top actions/targets to keep per concept
    
    Returns:
        Distilled model dictionary
    """
    print(f"Loading corpus from {corpus_path}...")
    qa = HolographicGeometricQA()
    qa.load_corpus(corpus_path)
    knowledge = qa.knowledge
    
    print(f"  Frames: {len(knowledge.frames)}")
    print(f"  Concepts: {len(knowledge.concepts)}")
    
    # Distill concepts
    print(f"\nDistilling concepts (min_freq={min_frequency}, top_k={top_k})...")
    distilled_concepts = {}
    skipped = 0
    
    for word, concept in knowledge.concepts.items():
        if concept.frequency < min_frequency:
            skipped += 1
            continue
        
        # Use compact format
        distilled_concepts[word] = distill_concept_compact(concept, top_k)
    
    print(f"  Distilled: {len(distilled_concepts)} concepts")
    print(f"  Skipped (low freq): {skipped}")
    
    # Extract morphology equivalences
    print("\nExtracting morphology...")
    morphology = {}
    if hasattr(knowledge, 'morphology') and knowledge.morphology:
        for canonical, equivalents in knowledge.morphology.equivalence_classes.items():
            if len(equivalents) > 1:
                morphology[canonical] = list(equivalents)
    print(f"  Morphology clusters: {len(morphology)}")
    
    # Extract relationship graph (concept -> concept edges)
    print("\nBuilding relationship graph...")
    relationships = {}
    for word, dc in distilled_concepts.items():
        edges = {}
        # dc is now a list: [phi_dir, freq, i, m, r, actions, targets]
        actions = dc[5]  # [[action, count], ...]
        targets = dc[6]  # [[target, count], ...]
        for action, count in actions:
            if action in distilled_concepts:
                edges[action] = edges.get(action, 0) + count
        for target, count in targets:
            if target in distilled_concepts:
                edges[target] = edges.get(target, 0) + count
        if edges:
            # Keep top relationships as list of [word, weight]
            top_edges = sorted(edges.items(), key=lambda x: -x[1])[:top_k]
            relationships[word] = top_edges
    print(f"  Concepts with relationships: {len(relationships)}")
    
    # Compute statistics
    # dc format: [phi_dir, freq, i, m, r, actions, targets]
    all_phi = [dc[0] for dc in distilled_concepts.values()]
    avg_phi = sum(all_phi) / len(all_phi) if all_phi else 0
    
    # Build distilled model
    model = {
        'version': '1.0',
        'format': 'compact',
        'schema': ['phi_dir', 'freq', 'i_count', 'm_count', 'r_count', 'actions', 'targets'],
        'source_corpus': corpus_path,
        'source_frames': len(knowledge.frames),
        'statistics': {
            'total_concepts': len(distilled_concepts),
            'morphology_clusters': len(morphology),
            'avg_phi_direction': avg_phi,
        },
        'concepts': distilled_concepts,
        'morphology': morphology,
        'relationships': relationships,
    }
    
    return model


def save_distilled(model: Dict, output_path: str):
    """Save distilled model to JSON (compact, no indent)."""
    with open(output_path, 'w') as f:
        json.dump(model, f, separators=(',', ':'))  # No whitespace
    
    # Calculate size
    size_bytes = Path(output_path).stat().st_size
    size_kb = size_bytes / 1024
    print(f"\nSaved to {output_path}")
    print(f"  Size: {size_kb:.1f} KB")


def compare_sizes(corpus_path: str, distilled_path: str):
    """Compare sizes of corpus vs distilled model."""
    corpus_size = Path(corpus_path).stat().st_size / 1024
    distilled_size = Path(distilled_path).stat().st_size / 1024
    ratio = corpus_size / distilled_size if distilled_size > 0 else 0
    
    print("\n" + "=" * 50)
    print("SIZE COMPARISON")
    print("=" * 50)
    print(f"  Full corpus:     {corpus_size:.1f} KB")
    print(f"  Distilled model: {distilled_size:.1f} KB")
    print(f"  Compression:     {ratio:.1f}x smaller")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Distill corpus to pure concepts')
    parser.add_argument('--corpus', default='truthspace_lcm/corpus_self_improved.json',
                       help='Path to corpus file')
    parser.add_argument('--output', default='truthspace_lcm/concepts_distilled.json',
                       help='Output path for distilled model')
    parser.add_argument('--min-freq', type=int, default=3,
                       help='Minimum frequency to include concept')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Top K actions/targets to keep per concept')
    
    args = parser.parse_args()
    
    # Distill
    model = distill_corpus(args.corpus, args.min_freq, args.top_k)
    
    # Save
    save_distilled(model, args.output)
    
    # Compare sizes
    compare_sizes(args.corpus, args.output)
    
    print("\n✓ Distillation complete!")
    print("\nThe distilled model contains:")
    print(f"  - {model['statistics']['total_concepts']} concepts with geometric properties")
    print(f"  - {model['statistics']['morphology_clusters']} morphology clusters")
    print(f"  - Relationship graph for concept navigation")
    print("\nNo text storage - pure geometric structure!")


if __name__ == '__main__':
    main()
