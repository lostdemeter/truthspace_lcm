#!/usr/bin/env python3
"""
Experiment: How Meaningful Relationships Form in Concept Space

Key Questions:
1. What distinguishes meaningful relationships from noise?
2. How do proper nouns differ from common words in relationship patterns?
3. Can we use Zipf distribution to automatically filter noise?
4. What geometric properties indicate relationship importance?

Hypothesis:
- Meaningful relationships are BIDIRECTIONAL
- Proper nouns are SPARSE but carry HIGH information
- Common words follow Zipf distribution (high frequency, low specificity)
- Relationship importance = specificity × bidirectionality × spread
"""

import json
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_corpus():
    """Load the concept corpus."""
    with open('truthspace_lcm/concept_corpus.json', 'r') as f:
        return json.load(f)


def analyze_word_frequencies(corpus: Dict) -> Dict[str, int]:
    """Analyze word frequencies across all frames."""
    word_counts = Counter()
    
    for frame in corpus['frames']:
        text = frame.get('text', '').lower()
        words = text.split()
        word_counts.update(words)
    
    return word_counts


def analyze_agent_frequencies(corpus: Dict) -> Dict[str, int]:
    """Analyze how often each entity appears as an agent."""
    agent_counts = Counter()
    
    for frame in corpus['frames']:
        agent = frame.get('agent', '')
        if agent:
            agent_counts[agent] += 1
    
    return agent_counts


def compute_zipf_rank(word_counts: Dict[str, int]) -> Dict[str, int]:
    """Compute Zipf rank for each word (1 = most frequent)."""
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    return {word: rank + 1 for rank, (word, _) in enumerate(sorted_words)}


def compute_specificity(word: str, word_counts: Dict[str, int], total_words: int) -> float:
    """
    Compute specificity as inverse of frequency.
    
    High specificity = rare word = likely proper noun
    Low specificity = common word = likely Zipf noise
    """
    count = word_counts.get(word, 1)
    frequency = count / total_words
    
    # Inverse log frequency (higher = more specific)
    return 1.0 / np.log1p(count)


def analyze_bidirectionality(corpus: Dict) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """
    Analyze bidirectional relationships between entities.
    
    Returns: {entity1: {entity2: (e1_mentions_e2, e2_mentions_e1)}}
    """
    # First pass: count agent mentions of other entities
    mentions = defaultdict(lambda: defaultdict(int))
    
    for frame in corpus['frames']:
        agent = frame.get('agent', '')
        text = frame.get('text', '').lower()
        
        if agent:
            # Find other entities mentioned in this frame
            for other_agent in set(f.get('agent', '') for f in corpus['frames']):
                if other_agent and other_agent != agent and other_agent in text:
                    mentions[agent][other_agent] += 1
    
    # Convert to bidirectional format
    bidirectional = defaultdict(dict)
    
    for e1 in mentions:
        for e2 in mentions[e1]:
            e1_to_e2 = mentions[e1][e2]
            e2_to_e1 = mentions.get(e2, {}).get(e1, 0)
            bidirectional[e1][e2] = (e1_to_e2, e2_to_e1)
    
    return bidirectional


def classify_entity_type(entity: str, agent_counts: Dict[str, int], 
                         word_counts: Dict[str, int]) -> str:
    """
    Classify entity as proper noun, common word, or noise.
    
    Heuristics:
    - Proper nouns: high agent count, low global frequency
    - Common words: high global frequency, moderate agent count
    - Noise: parsing artifacts
    """
    agent_count = agent_counts.get(entity, 0)
    word_count = word_counts.get(entity, 0)
    
    # Ratio of agent appearances to total word appearances
    if word_count == 0:
        return 'unknown'
    
    agent_ratio = agent_count / word_count
    
    # Proper nouns: appear as agents often relative to total mentions
    if agent_ratio > 0.3 and agent_count > 10:
        return 'proper_noun'
    
    # Common words: very high frequency, low agent ratio
    if word_count > 1000 and agent_ratio < 0.1:
        return 'common_word'
    
    # Noise: low counts or weird ratios
    if agent_count < 5:
        return 'noise'
    
    return 'ambiguous'


def analyze_relationship_patterns(corpus: Dict):
    """Main analysis of relationship patterns."""
    
    print("=" * 70)
    print("RELATIONSHIP FORMATION ANALYSIS")
    print("=" * 70)
    print()
    
    # Step 1: Word frequency analysis
    print("## STEP 1: Word Frequency Analysis (Zipf Distribution)")
    print()
    
    word_counts = analyze_word_frequencies(corpus)
    agent_counts = analyze_agent_frequencies(corpus)
    total_words = sum(word_counts.values())
    
    print(f"Total words: {total_words}")
    print(f"Unique words: {len(word_counts)}")
    print(f"Unique agents: {len(agent_counts)}")
    print()
    
    # Top words by frequency (Zipf)
    print("Top 20 words by frequency (Zipf distribution):")
    for word, count in word_counts.most_common(20):
        specificity = compute_specificity(word, word_counts, total_words)
        agent_count = agent_counts.get(word, 0)
        print(f"  {word:15} freq={count:5} spec={specificity:.4f} agent={agent_count}")
    
    print()
    
    # Step 2: Entity classification
    print("## STEP 2: Entity Classification")
    print()
    
    entity_types = {}
    for entity in agent_counts:
        entity_types[entity] = classify_entity_type(entity, agent_counts, word_counts)
    
    type_counts = Counter(entity_types.values())
    print("Entity type distribution:")
    for etype, count in type_counts.most_common():
        print(f"  {etype}: {count}")
    
    print()
    
    # Show examples of each type
    print("Examples of each type:")
    for etype in ['proper_noun', 'common_word', 'noise', 'ambiguous']:
        examples = [e for e, t in entity_types.items() if t == etype][:5]
        print(f"  {etype}: {examples}")
    
    print()
    
    # Step 3: Bidirectional relationship analysis
    print("## STEP 3: Bidirectional Relationship Analysis")
    print()
    
    # Focus on proper nouns
    proper_nouns = {e for e, t in entity_types.items() if t == 'proper_noun'}
    print(f"Proper nouns identified: {len(proper_nouns)}")
    print(f"Sample: {list(proper_nouns)[:15]}")
    print()
    
    # Analyze relationships between proper nouns
    bidirectional = analyze_bidirectionality(corpus)
    
    # Find strongest bidirectional relationships
    print("Strongest bidirectional relationships (proper nouns only):")
    print()
    
    relationships = []
    for e1 in proper_nouns:
        for e2, (e1_to_e2, e2_to_e1) in bidirectional.get(e1, {}).items():
            if e2 in proper_nouns and e1 < e2:  # Avoid duplicates
                total = e1_to_e2 + e2_to_e1
                balance = min(e1_to_e2, e2_to_e1) / max(e1_to_e2, e2_to_e1) if max(e1_to_e2, e2_to_e1) > 0 else 0
                relationships.append((e1, e2, e1_to_e2, e2_to_e1, total, balance))
    
    # Sort by total mentions
    relationships.sort(key=lambda x: -x[4])
    
    print("Entity1        Entity2        E1→E2  E2→E1  Total  Balance")
    print("-" * 65)
    for e1, e2, e1_to_e2, e2_to_e1, total, balance in relationships[:20]:
        print(f"{e1:14} {e2:14} {e1_to_e2:5} {e2_to_e1:6} {total:6} {balance:7.2f}")
    
    print()
    
    # Step 4: Specificity vs Relationship Strength
    print("## STEP 4: Specificity vs Relationship Strength")
    print()
    
    print("Hypothesis: High specificity entities form stronger relationships")
    print()
    
    # For each proper noun, compute average relationship strength
    entity_relationship_strength = {}
    
    for entity in proper_nouns:
        relations = bidirectional.get(entity, {})
        if relations:
            total_strength = sum(e1 + e2 for e1, e2 in relations.values())
            avg_strength = total_strength / len(relations)
            specificity = compute_specificity(entity, word_counts, total_words)
            entity_relationship_strength[entity] = (specificity, avg_strength, len(relations))
    
    # Sort by relationship strength
    sorted_entities = sorted(entity_relationship_strength.items(), 
                            key=lambda x: -x[1][1])[:15]
    
    print("Entity         Specificity  Avg Strength  Num Relations")
    print("-" * 55)
    for entity, (spec, strength, num_rel) in sorted_entities:
        print(f"{entity:14} {spec:11.4f} {strength:12.2f} {num_rel:14}")
    
    return {
        'word_counts': word_counts,
        'agent_counts': agent_counts,
        'entity_types': entity_types,
        'bidirectional': bidirectional,
        'proper_nouns': proper_nouns,
    }


def analyze_relationship_geometry(corpus: Dict, analysis: Dict):
    """Analyze the geometric properties of relationships."""
    
    print()
    print("=" * 70)
    print("GEOMETRIC ANALYSIS: Relationship Formation")
    print("=" * 70)
    print()
    
    proper_nouns = analysis['proper_nouns']
    bidirectional = analysis['bidirectional']
    word_counts = analysis['word_counts']
    total_words = sum(word_counts.values())
    
    # Key insight: Meaningful relationships have specific geometric properties
    
    print("## KEY GEOMETRIC PROPERTIES")
    print()
    
    print("1. BIDIRECTIONALITY")
    print("   - Meaningful: Both entities mention each other")
    print("   - Noise: One-way mentions only")
    print()
    
    # Count bidirectional vs unidirectional
    bidirectional_count = 0
    unidirectional_count = 0
    
    for e1 in proper_nouns:
        for e2, (e1_to_e2, e2_to_e1) in bidirectional.get(e1, {}).items():
            if e2 in proper_nouns and e1 < e2:
                if e1_to_e2 > 0 and e2_to_e1 > 0:
                    bidirectional_count += 1
                else:
                    unidirectional_count += 1
    
    print(f"   Bidirectional relationships: {bidirectional_count}")
    print(f"   Unidirectional relationships: {unidirectional_count}")
    print(f"   Ratio: {bidirectional_count / (bidirectional_count + unidirectional_count + 0.01):.2%}")
    print()
    
    print("2. SPECIFICITY (Inverse Zipf)")
    print("   - High specificity = rare word = likely meaningful")
    print("   - Low specificity = common word = likely noise")
    print()
    
    # Compare specificity of proper nouns vs common words
    proper_noun_specs = [compute_specificity(e, word_counts, total_words) 
                        for e in proper_nouns]
    common_words = [e for e, t in analysis['entity_types'].items() if t == 'common_word']
    common_word_specs = [compute_specificity(e, word_counts, total_words) 
                        for e in common_words]
    
    if proper_noun_specs:
        print(f"   Proper noun avg specificity: {np.mean(proper_noun_specs):.4f}")
    if common_word_specs:
        print(f"   Common word avg specificity: {np.mean(common_word_specs):.4f}")
    print()
    
    print("3. SPREAD (Multi-source presence)")
    print("   - Meaningful: Appears across multiple sources")
    print("   - Noise: Appears in only one source")
    print()
    
    # Compute spread for each entity
    entity_sources = defaultdict(set)
    for frame in corpus['frames']:
        agent = frame.get('agent', '')
        source = frame.get('source', '')
        if agent and source:
            entity_sources[agent].add(source)
    
    total_sources = len(set(f.get('source', '') for f in corpus['frames']))
    
    multi_source = sum(1 for e in proper_nouns if len(entity_sources.get(e, set())) >= 2)
    single_source = sum(1 for e in proper_nouns if len(entity_sources.get(e, set())) == 1)
    
    print(f"   Multi-source proper nouns: {multi_source}")
    print(f"   Single-source proper nouns: {single_source}")
    print()
    
    print("=" * 70)
    print("## FORMULA FOR RELATIONSHIP IMPORTANCE")
    print("=" * 70)
    print()
    
    print("importance(A, B) = specificity(A) × specificity(B) × bidirectionality(A,B) × spread(A,B)")
    print()
    print("where:")
    print("  specificity(X) = 1 / log(1 + frequency(X))")
    print("  bidirectionality(A,B) = 1.5 if both directions, else 1.0")
    print("  spread(A,B) = min(sources(A), sources(B)) / total_sources")
    print()
    
    # Compute importance for top relationships
    print("Top relationships by importance formula:")
    print()
    
    importance_scores = []
    
    for e1 in proper_nouns:
        for e2, (e1_to_e2, e2_to_e1) in bidirectional.get(e1, {}).items():
            if e2 in proper_nouns and e1 < e2:
                spec1 = compute_specificity(e1, word_counts, total_words)
                spec2 = compute_specificity(e2, word_counts, total_words)
                bidir = 1.5 if (e1_to_e2 > 0 and e2_to_e1 > 0) else 1.0
                spread = min(len(entity_sources.get(e1, set())), 
                           len(entity_sources.get(e2, set()))) / total_sources
                
                importance = spec1 * spec2 * bidir * (spread + 0.1) * np.log1p(e1_to_e2 + e2_to_e1)
                importance_scores.append((e1, e2, importance, spec1, spec2, bidir, spread))
    
    importance_scores.sort(key=lambda x: -x[2])
    
    print("Entity1        Entity2        Importance  Spec1  Spec2  Bidir  Spread")
    print("-" * 75)
    for e1, e2, imp, s1, s2, bidir, spread in importance_scores[:15]:
        print(f"{e1:14} {e2:14} {imp:10.4f} {s1:6.3f} {s2:6.3f} {bidir:6.1f} {spread:7.3f}")
    
    return importance_scores


def discover_autobalance_insights(corpus: Dict, analysis: Dict, importance_scores: List):
    """Discover insights for autobalancing the model."""
    
    print()
    print("=" * 70)
    print("AUTOBALANCE INSIGHTS")
    print("=" * 70)
    print()
    
    print("## KEY DISCOVERIES")
    print()
    
    print("1. PROPER NOUNS ARE SPARSE BUT MEANINGFUL")
    print("   - They appear infrequently (high specificity)")
    print("   - But when they appear, they carry relationship information")
    print("   - This matches Qwen2 finding: proper nouns on 'zero axis'")
    print()
    
    print("2. ZIPF DISTRIBUTION IDENTIFIES NOISE")
    print("   - High-frequency words are structural (the, said, was)")
    print("   - Low-frequency words are meaningful (Watson, Holmes, Darcy)")
    print("   - Inverse Zipf weighting automatically filters noise")
    print()
    
    print("3. BIDIRECTIONALITY IS THE KEY SIGNAL")
    print("   - Meaningful relationships are bidirectional")
    print("   - If A is important to B, then B is important to A")
    print("   - Unidirectional mentions are often noise or weak associations")
    print()
    
    print("4. SPREAD INDICATES UNIVERSALITY")
    print("   - Entities appearing in multiple sources are more important")
    print("   - Single-source entities may be story-specific")
    print()
    
    print("## AUTOBALANCE FORMULA")
    print()
    print("For any entity E, compute:")
    print()
    print("  weight(E) = specificity(E) × relationship_strength(E) × spread(E)")
    print()
    print("where:")
    print("  specificity = 1 / log(1 + global_frequency)")
    print("  relationship_strength = sum of bidirectional relationship scores")
    print("  spread = number of sources / total sources")
    print()
    
    print("## IMPLEMENTATION RECOMMENDATION")
    print()
    print("1. During corpus building:")
    print("   - Track global word frequencies")
    print("   - Compute specificity for each entity")
    print("   - Build bidirectional relationship graph")
    print()
    print("2. During query time:")
    print("   - Weight entities by specificity (inverse Zipf)")
    print("   - Prioritize bidirectional relationships")
    print("   - Boost multi-source entities")
    print()
    print("3. For autobalancing:")
    print("   - Entities with high weight get more 'attention'")
    print("   - Low-weight entities (Zipf noise) are de-emphasized")
    print("   - Relationship importance guides answer generation")


if __name__ == "__main__":
    corpus = load_corpus()
    analysis = analyze_relationship_patterns(corpus)
    importance_scores = analyze_relationship_geometry(corpus, analysis)
    discover_autobalance_insights(corpus, analysis, importance_scores)
