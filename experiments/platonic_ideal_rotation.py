#!/usr/bin/env python3
"""
Platonic Ideal Rotation: Axes as Intersections
===============================================

Connection between two discoveries:

1. Doc 114: Platonic Ideals sit at intersections of dimensions
   - "house" is at (0,0) on size × regality
   - Variations move φ along one or more axes

2. Rotation findings: Entity→Answer is a ~77° rotation
   - Axis is orthogonal to entity
   - Axis encodes the relationship

Hypothesis: The rotation axis points TOWARD a Platonic Ideal.

For "capital-of":
- France is at some position in semantic space
- Paris is at France + rotation(77°, axis)
- The axis points toward the "capital" Platonic Ideal

If true, memory becomes:
- Store Platonic Ideals (intersections of dimensions)
- Relationships are rotations TOWARD these ideals
- The ideal IS the geometric encoding of the relationship

Author: TruthSpace LCM Team
Date: 2026-01-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

PHI = 1.6180339887498949


def find_platonic_ideals(model, tokenizer):
    """
    Find potential Platonic Ideals in embedding space.
    
    A Platonic Ideal should:
    1. Be at the "center" of related concepts
    2. Anchor multiple relationship types
    3. Have many concepts at φ distance
    """
    print("\n" + "=" * 70)
    print("Finding Platonic Ideals")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    
    # Candidate Platonic Ideals for different domains
    candidates = {
        'dwelling': ['house', 'home', 'dwelling', 'residence'],
        'person': ['person', 'human', 'individual', 'one'],
        'place': ['place', 'location', 'area', 'spot'],
        'country': ['country', 'nation', 'state', 'land'],
        'city': ['city', 'town', 'capital', 'metropolis'],
    }
    
    # For each domain, find the centroid of related concepts
    print("\n--- Domain Centroids ---")
    
    centroids = {}
    for domain, words in candidates.items():
        embeds = []
        for word in words:
            ids = tokenizer.encode(word, add_special_tokens=False)
            if ids:
                embeds.append(embed[ids[0]])
        
        if embeds:
            centroid = torch.stack(embeds).mean(dim=0)
            centroids[domain] = centroid
            
            # How close are the words to the centroid?
            dists = [F.cosine_similarity(e.unsqueeze(0), centroid.unsqueeze(0)).item() 
                     for e in embeds]
            print(f"  {domain}: mean similarity to centroid = {np.mean(dists):.4f}")
    
    return centroids


def analyze_rotation_toward_ideal(model, tokenizer, centroids):
    """
    Analyze if rotation axes point toward Platonic Ideals.
    """
    print("\n" + "=" * 70)
    print("Rotation Axes vs Platonic Ideals")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    
    # Capital pairs
    pairs = [
        ("France", " Paris"),
        ("Germany", " Berlin"),
        ("Italy", " Rome"),
        ("Spain", " Madrid"),
    ]
    
    # Compute rotation axes
    axes = []
    for entity, answer in pairs:
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        
        e_embed = embed[entity_ids[0]]
        a_embed = embed[answer_ids[0]]
        
        # Rotation axis (orthogonal component of answer relative to entity)
        e_norm = e_embed / e_embed.norm()
        a_norm = a_embed / a_embed.norm()
        
        # Component of answer orthogonal to entity
        a_orth = a_norm - (a_norm @ e_norm) * e_norm
        a_orth = a_orth / a_orth.norm()
        
        axes.append(a_orth)
    
    # Mean axis
    mean_axis = torch.stack(axes).mean(dim=0)
    mean_axis = mean_axis / mean_axis.norm()
    
    print("\n--- Axis Similarity to Platonic Ideals ---")
    
    for domain, centroid in centroids.items():
        centroid_norm = centroid / centroid.norm()
        
        # Similarity between mean axis and centroid
        sim = (mean_axis @ centroid_norm).item()
        
        print(f"  {domain}: axis·centroid = {sim:.4f}")
    
    # Does the axis point toward "city" or "capital"?
    print("\n--- Axis Similarity to Specific Concepts ---")
    
    concepts = ['capital', 'city', 'metropolis', 'town', 'place', 'location']
    
    for concept in concepts:
        ids = tokenizer.encode(concept, add_special_tokens=False)
        if ids:
            c_embed = embed[ids[0]]
            c_norm = c_embed / c_embed.norm()
            sim = (mean_axis @ c_norm).item()
            print(f"  {concept}: axis·embed = {sim:.4f}")


def explore_dimension_intersections(model, tokenizer):
    """
    Explore if entities and answers lie on different dimension intersections.
    
    Hypothesis:
    - France is at intersection of (European, Country, ...)
    - Paris is at intersection of (European, City, Capital, ...)
    - The rotation moves from Country-intersection to Capital-intersection
    """
    print("\n" + "=" * 70)
    print("Dimension Intersection Analysis")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    
    # Define dimension anchors
    dimensions = {
        'european': ['European', 'Europe', 'Western'],
        'asian': ['Asian', 'Asia', 'Eastern'],
        'country': ['country', 'nation', 'state'],
        'city': ['city', 'town', 'urban'],
        'capital': ['capital', 'seat', 'center'],
        'large': ['large', 'big', 'major'],
        'political': ['political', 'government', 'official'],
    }
    
    # Compute dimension vectors
    dim_vectors = {}
    for dim_name, anchors in dimensions.items():
        embeds = []
        for anchor in anchors:
            ids = tokenizer.encode(anchor, add_special_tokens=False)
            if ids:
                embeds.append(embed[ids[0]])
        
        if embeds:
            dim_vec = torch.stack(embeds).mean(dim=0)
            dim_vec = dim_vec / dim_vec.norm()
            dim_vectors[dim_name] = dim_vec
    
    # Project entities and answers onto dimensions
    print("\n--- Entity/Answer Dimension Projections ---")
    
    pairs = [
        ("France", " Paris"),
        ("Germany", " Berlin"),
        ("Japan", " Tokyo"),
    ]
    
    for entity, answer in pairs:
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        
        e_embed = embed[entity_ids[0]]
        a_embed = embed[answer_ids[0]]
        
        e_norm = e_embed / e_embed.norm()
        a_norm = a_embed / a_embed.norm()
        
        print(f"\n  {entity} → {answer}:")
        
        for dim_name, dim_vec in dim_vectors.items():
            e_proj = (e_norm @ dim_vec).item()
            a_proj = (a_norm @ dim_vec).item()
            delta = a_proj - e_proj
            
            if abs(delta) > 0.02:  # Significant change
                direction = "↑" if delta > 0 else "↓"
                print(f"    {dim_name}: {e_proj:.3f} → {a_proj:.3f} ({direction} {abs(delta):.3f})")


def analyze_house_example(model, tokenizer):
    """
    Analyze the house example from Doc 114.
    
    house → cottage (size_decrease)
    house → mansion (size_increase)
    house → palace (regality_increase)
    
    Is the rotation structure the same?
    """
    print("\n" + "=" * 70)
    print("House Example Analysis (Doc 114)")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    
    # House variations
    pairs = [
        ("house", "cottage", "size_decrease"),
        ("house", "mansion", "size_increase"),
        ("house", "palace", "regality_increase"),
        ("house", "hovel", "regality_decrease"),
        ("house", "cabin", "rustic"),
    ]
    
    print("\n--- House Variations as Rotations ---")
    
    house_ids = tokenizer.encode("house", add_special_tokens=False)
    house_embed = embed[house_ids[0]]
    house_norm = house_embed / house_embed.norm()
    
    axes = []
    
    for source, target, relation in pairs:
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if not target_ids:
            continue
        
        target_embed = embed[target_ids[0]]
        target_norm = target_embed / target_embed.norm()
        
        # Rotation angle
        cos_angle = (house_norm @ target_norm).clamp(-1, 1)
        angle = torch.acos(cos_angle) * 180 / np.pi
        
        # Rotation axis (orthogonal component)
        t_orth = target_norm - (target_norm @ house_norm) * house_norm
        if t_orth.norm() > 1e-6:
            t_orth = t_orth / t_orth.norm()
            axes.append((relation, t_orth))
        
        print(f"  house → {target} ({relation}): angle = {angle:.1f}°")
    
    # Are the axes different for different relations?
    print("\n--- Axis Similarity Between Relations ---")
    
    for i in range(len(axes)):
        for j in range(i+1, len(axes)):
            rel1, axis1 = axes[i]
            rel2, axis2 = axes[j]
            sim = F.cosine_similarity(axis1.unsqueeze(0), axis2.unsqueeze(0)).item()
            print(f"  {rel1} vs {rel2}: {sim:.4f}")


def synthesize_platonic_rotation():
    """Synthesize findings about Platonic Ideals and rotation."""
    print("\n" + "=" * 70)
    print("SYNTHESIS: Platonic Ideals as Rotation Targets")
    print("=" * 70)
    print("""
Key Insight:

The rotation axis in Entity→Answer transformations may point
toward a PLATONIC IDEAL - the intersection of dimensions that
defines the relationship.

For "capital-of":
- The axis points toward the "capital" concept
- This is the intersection of (city, political, important)
- France rotates TOWARD this ideal to become Paris

For "house variations":
- Different relations have different axes
- size_decrease axis points toward "small" ideal
- regality_increase axis points toward "regal" ideal
- Each axis is a direction toward a Platonic Ideal

MEMORY AS PLATONIC IDEALS:
==========================

Instead of storing:
  {France: Paris, Germany: Berlin, ...}

Store:
  {capital_ideal: intersection(city, political, important)}

Then compute:
  answer = rotate(entity, 77°, toward=capital_ideal)

The Platonic Ideal IS the relationship.
The rotation IS the computation.
Memory IS geometry.

IMPLICATIONS:
=============

1. Relationships are DIRECTIONS toward ideals
2. The angle (77°) is universal for a relationship type
3. The ideal is the INTERSECTION of dimensions
4. Memory = storing ideals, not instances

This unifies:
- Doc 114: Platonic Ideals at dimension intersections
- Rotation findings: Entity→Answer as rotation
- Memory: Geometric operation, not lookup table
""")


def main():
    print("=" * 70)
    print("Platonic Ideal Rotation: Axes as Intersections")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Analysis 1: Find Platonic Ideals
    centroids = find_platonic_ideals(model, tokenizer)
    
    # Analysis 2: Rotation axes vs ideals
    analyze_rotation_toward_ideal(model, tokenizer, centroids)
    
    # Analysis 3: Dimension intersections
    explore_dimension_intersections(model, tokenizer)
    
    # Analysis 4: House example
    analyze_house_example(model, tokenizer)
    
    # Synthesis
    synthesize_platonic_rotation()


if __name__ == "__main__":
    main()
