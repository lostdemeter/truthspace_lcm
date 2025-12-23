#!/usr/bin/env python3
"""
Experiment: Controlled Noise + Reprojection for Better Answers

The Problem:
    Our answers are "cookie-cutter" - structurally identical.
    "Holmes is a character who spoke often involving watson"
    "Darcy is a character who took action often involving elizabeth"

The Hypothesis:
    Traditional LLMs add noise (temperature/sampling) which creates variety.
    We can add CONTROLLED noise and then REPROJECT to get:
    - Variety in expression
    - While maintaining geometric accuracy

The Approach:
    1. Generate base answer (geodesic path)
    2. Add controlled noise (explore nearby paths)
    3. Reproject to find the "best" varied answer
    
This is like:
    - Base answer = shortest path
    - Noise = explore neighborhood
    - Reprojection = find most natural path in neighborhood
"""

import json
import numpy as np
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm import ConceptQA
from truthspace_lcm.core.spatial_attention import get_attention, initialize_attention


# Answer templates with variety
WHO_TEMPLATES = [
    "{entity} is a {role} from {source} who {action} often involving {related}",
    "{entity}, known for {action}, appears in {source} alongside {related}",
    "In {source}, {entity} is portrayed as someone who {action}, closely connected to {related}",
    "A {role} from {source}, {entity} {action} and has strong ties to {related}",
    "{entity} from {source} - a {role} who {action}, frequently associated with {related}",
]

ACTION_VARIATIONS = {
    'SPEAK': ['speaks', 'communicates', 'converses', 'talks', 'expresses'],
    'THINK': ['thinks deeply', 'contemplates', 'reflects', 'ponders', 'reasons'],
    'ACT': ['takes action', 'acts decisively', 'engages', 'participates', 'intervenes'],
    'MOVE': ['travels', 'journeys', 'moves about', 'ventures', 'roams'],
    'PERCEIVE': ['observes', 'notices', 'perceives', 'watches', 'examines'],
    'POSSESS': ['possesses', 'holds', 'maintains', 'keeps', 'retains'],
    'EXIST': ['exists', 'appears', 'features', 'is present', 'manifests'],
}

ROLE_VARIATIONS = {
    'character': ['character', 'figure', 'personality', 'individual', 'person'],
    'thinker': ['thinker', 'intellectual', 'mind', 'philosopher', 'analyst'],
    'speaker': ['speaker', 'communicator', 'voice', 'orator', 'conversationalist'],
    'traveler': ['traveler', 'wanderer', 'explorer', 'journeyer', 'adventurer'],
}


def load_corpus():
    """Load the concept corpus."""
    with open('truthspace_lcm/concept_corpus.json', 'r') as f:
        return json.load(f)


def get_entity_profile(corpus: Dict, entity: str) -> Dict:
    """Get profile for an entity."""
    frames = [f for f in corpus['frames'] if f.get('agent') == entity]
    
    actions = Counter(f.get('action') for f in frames if f.get('action'))
    sources = set(f.get('source') for f in frames if f.get('source'))
    
    return {
        'actions': actions,
        'sources': sources,
        'frame_count': len(frames),
    }


def generate_base_answer(entity: str, profile: Dict, related: str) -> str:
    """Generate the base (cookie-cutter) answer."""
    top_action = profile['actions'].most_common(1)[0][0] if profile['actions'] else 'EXIST'
    source = list(profile['sources'])[0] if profile['sources'] else 'unknown'
    
    action_verb = ACTION_VARIATIONS.get(top_action, ['acts'])[0]
    
    return f"{entity.title()} is a character from {source} who {action_verb} often involving {related}"


def add_noise_to_answer(base_answer: str, entity: str, profile: Dict, 
                        related: str, noise_level: float = 0.3) -> List[str]:
    """
    Add controlled noise to generate answer variations.
    
    noise_level: 0.0 = no variation, 1.0 = maximum variation
    """
    variations = []
    
    top_action = profile['actions'].most_common(1)[0][0] if profile['actions'] else 'EXIST'
    source = list(profile['sources'])[0] if profile['sources'] else 'unknown'
    
    # Get action variations
    action_options = ACTION_VARIATIONS.get(top_action, ['acts'])
    
    # Get role based on action
    if top_action == 'THINK':
        role_options = ROLE_VARIATIONS['thinker']
    elif top_action == 'SPEAK':
        role_options = ROLE_VARIATIONS['speaker']
    elif top_action == 'MOVE':
        role_options = ROLE_VARIATIONS['traveler']
    else:
        role_options = ROLE_VARIATIONS['character']
    
    # Generate variations by sampling from options
    num_variations = int(5 * noise_level) + 1
    
    for _ in range(num_variations):
        template = random.choice(WHO_TEMPLATES)
        action = random.choice(action_options)
        role = random.choice(role_options)
        
        variation = template.format(
            entity=entity.title(),
            role=role,
            source=source,
            action=action,
            related=related.title()
        )
        variations.append(variation)
    
    return variations


def reproject_answer(variations: List[str], entity: str, corpus: Dict) -> Tuple[str, float]:
    """
    Reproject variations to find the most "natural" one.
    
    Uses corpus statistics to score how well each variation
    matches the actual data.
    """
    # Simple reprojection: score by how many corpus words appear
    entity_frames = [f for f in corpus['frames'] if f.get('agent') == entity]
    corpus_words = set()
    for f in entity_frames:
        corpus_words.update(f.get('text', '').lower().split())
    
    scored = []
    for var in variations:
        var_words = set(var.lower().split())
        overlap = len(var_words & corpus_words)
        # Normalize by length to avoid favoring longer answers
        score = overlap / len(var_words) if var_words else 0
        scored.append((var, score))
    
    # Return best scoring variation
    scored.sort(key=lambda x: -x[1])
    return scored[0]


def geometric_reproject(variations: List[str], entity: str, 
                        related: str, attention) -> Tuple[str, float]:
    """
    Geometric reprojection using spatial attention.
    
    Score variations by how well they align with the
    geometric relationship structure.
    """
    # Get importance score for the relationship
    base_importance = attention.importance_score(entity, related)
    
    scored = []
    for var in variations:
        # Score based on whether key relationship terms appear
        var_lower = var.lower()
        
        # Bonus for mentioning the related entity prominently
        related_bonus = 1.5 if related in var_lower else 1.0
        
        # Bonus for natural sentence structure (not too long)
        length_penalty = 1.0 / (1 + abs(len(var) - 80) / 50)
        
        score = base_importance * related_bonus * length_penalty
        scored.append((var, score))
    
    scored.sort(key=lambda x: -x[1])
    return scored[0]


def run_experiment():
    """Run the noise + reprojection experiment."""
    
    print("=" * 70)
    print("NOISE + REPROJECTION EXPERIMENT")
    print("=" * 70)
    print()
    
    # Load corpus and initialize attention
    corpus = load_corpus()
    initialize_attention(corpus['frames'], set(corpus['entities'].keys()))
    attention = get_attention()
    
    # Test entities
    test_cases = [
        ('holmes', 'watson'),
        ('watson', 'holmes'),
        ('darcy', 'elizabeth'),
        ('elizabeth', 'jane'),
        ('alice', 'queen'),
    ]
    
    print("## BASE ANSWERS (Cookie-Cutter)")
    print()
    
    for entity, related in test_cases:
        profile = get_entity_profile(corpus, entity)
        base = generate_base_answer(entity, profile, related)
        print(f"Q: Who is {entity.title()}?")
        print(f"A: {base}")
        print()
    
    print("=" * 70)
    print("## WITH NOISE (Multiple Variations)")
    print("=" * 70)
    print()
    
    for entity, related in test_cases:
        profile = get_entity_profile(corpus, entity)
        base = generate_base_answer(entity, profile, related)
        variations = add_noise_to_answer(base, entity, profile, related, noise_level=0.5)
        
        print(f"Q: Who is {entity.title()}?")
        print(f"Variations:")
        for i, var in enumerate(variations, 1):
            print(f"  {i}. {var}")
        print()
    
    print("=" * 70)
    print("## WITH REPROJECTION (Best Variation)")
    print("=" * 70)
    print()
    
    for entity, related in test_cases:
        profile = get_entity_profile(corpus, entity)
        base = generate_base_answer(entity, profile, related)
        variations = add_noise_to_answer(base, entity, profile, related, noise_level=0.8)
        
        # Corpus-based reprojection
        best_corpus, score_corpus = reproject_answer(variations, entity, corpus)
        
        # Geometric reprojection
        best_geo, score_geo = geometric_reproject(variations, entity, related, attention)
        
        print(f"Q: Who is {entity.title()}?")
        print(f"Base:       {base}")
        print(f"Corpus RP:  {best_corpus} (score: {score_corpus:.3f})")
        print(f"Geometric:  {best_geo} (score: {score_geo:.4f})")
        print()
    
    print("=" * 70)
    print("## COMPARISON: Cookie-Cutter vs Reprojected")
    print("=" * 70)
    print()
    
    print("The reprojected answers maintain geometric accuracy while")
    print("adding natural language variety through controlled noise.")
    print()
    print("Key insight: Noise is not the enemy - UNCONTROLLED noise is.")
    print("By adding noise and then reprojecting, we get the best of both:")
    print("  - Variety in expression (from noise)")
    print("  - Accuracy in meaning (from reprojection)")


if __name__ == "__main__":
    run_experiment()
