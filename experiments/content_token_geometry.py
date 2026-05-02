#!/usr/bin/env python3
"""
Content Token Geometry: Investigating the Zero Axis
=====================================================

Hypothesis: Content tokens (Paris, Einstein, etc.) have geometric structure
that follows φ-Zipf principles on the "zero axis".

Key observations to test:
1. Do proper nouns cluster in embedding space?
2. Is there a geometric relationship between France→Paris?
3. Can we predict content tokens from geometric position?

From Doc 039: φ^(-rank) weighting = Zipf weighting for ranking
From Doc 164: Signs encode semantics, Levels encode energy

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2


def analyze_embedding_structure(model, tokenizer):
    """Analyze the geometric structure of embeddings."""
    
    print("=" * 70)
    print("EMBEDDING STRUCTURE ANALYSIS")
    print("=" * 70)
    
    embeddings = model.model.embed_tokens.weight.data.float()
    print(f"\nEmbedding shape: {embeddings.shape}")
    
    # Compute levels (magnitude) and signs
    levels = torch.log(embeddings.abs().clamp(min=1e-10)) / np.log(PHI)
    signs = torch.sign(embeddings)
    
    # Statistics
    print(f"\nLevel statistics:")
    print(f"  Mean: {levels.mean().item():.2f}")
    print(f"  Std:  {levels.std().item():.2f}")
    print(f"  Min:  {levels.min().item():.2f}")
    print(f"  Max:  {levels.max().item():.2f}")
    
    return embeddings, levels, signs


def find_zero_axis_tokens(embeddings, tokenizer, threshold=0.1):
    """
    Find tokens that lie near the "zero axis" (low magnitude in many dimensions).
    
    Hypothesis: Proper nouns cluster here.
    """
    print("\n" + "=" * 70)
    print("ZERO AXIS ANALYSIS")
    print("=" * 70)
    
    # Compute per-token "zero-ness" (how many dimensions are near zero)
    near_zero = (embeddings.abs() < threshold).float().mean(dim=1)
    
    # Find tokens with high zero-ness
    top_zero_indices = near_zero.argsort(descending=True)[:50]
    
    print(f"\nTokens with most near-zero dimensions (threshold={threshold}):")
    for i, idx in enumerate(top_zero_indices[:20]):
        token = tokenizer.decode([idx.item()])
        zero_pct = near_zero[idx].item() * 100
        print(f"  {i+1:2d}. {repr(token):20s} ({zero_pct:.1f}% near-zero)")
    
    return near_zero


def analyze_content_tokens(embeddings, tokenizer):
    """Analyze the geometric position of known content tokens."""
    
    print("\n" + "=" * 70)
    print("CONTENT TOKEN ANALYSIS")
    print("=" * 70)
    
    # Content tokens to analyze
    content_pairs = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Japan", "Tokyo"),
        ("Einstein", "physics"),
        ("Shakespeare", "playwright"),
    ]
    
    results = []
    
    for entity, answer in content_pairs:
        # Get token IDs
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        
        # Get embeddings (use first token if multi-token)
        if entity_ids and answer_ids:
            entity_emb = embeddings[entity_ids[0]]
            answer_emb = embeddings[answer_ids[0]]
            
            # Compute relationship
            diff = answer_emb - entity_emb
            
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                entity_emb.unsqueeze(0), 
                answer_emb.unsqueeze(0)
            ).item()
            
            # Magnitude
            entity_mag = entity_emb.norm().item()
            answer_mag = answer_emb.norm().item()
            diff_mag = diff.norm().item()
            
            # Angle
            angle = np.arccos(np.clip(cos_sim, -1, 1)) * 180 / np.pi
            
            results.append({
                'entity': entity,
                'answer': answer,
                'cos_sim': cos_sim,
                'angle': angle,
                'entity_mag': entity_mag,
                'answer_mag': answer_mag,
                'diff_mag': diff_mag,
            })
            
            print(f"\n{entity} → {answer}:")
            print(f"  Cosine similarity: {cos_sim:.4f}")
            print(f"  Angle: {angle:.1f}°")
            print(f"  Entity magnitude: {entity_mag:.2f}")
            print(f"  Answer magnitude: {answer_mag:.2f}")
            print(f"  Difference magnitude: {diff_mag:.2f}")
    
    return results


def analyze_capital_relationship(embeddings, tokenizer):
    """
    Analyze if there's a consistent geometric transformation for capital-of.
    
    If France→Paris and Germany→Berlin share the same transformation,
    we could predict capitals geometrically.
    """
    print("\n" + "=" * 70)
    print("CAPITAL-OF RELATIONSHIP ANALYSIS")
    print("=" * 70)
    
    capitals = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
        ("Poland", "Warsaw"),
        ("Japan", "Tokyo"),
        ("China", "Beijing"),
        ("Brazil", "Brasilia"),
        ("Egypt", "Cairo"),
        ("Greece", "Athens"),
    ]
    
    # Collect transformation vectors
    transformations = []
    
    for country, capital in capitals:
        country_ids = tokenizer.encode(country, add_special_tokens=False)
        capital_ids = tokenizer.encode(capital, add_special_tokens=False)
        
        if country_ids and capital_ids:
            country_emb = embeddings[country_ids[0]]
            capital_emb = embeddings[capital_ids[0]]
            
            # Transformation vector
            transform = capital_emb - country_emb
            transformations.append({
                'country': country,
                'capital': capital,
                'transform': transform,
            })
    
    if len(transformations) < 2:
        print("Not enough data")
        return
    
    # Compute pairwise similarities between transformations
    print(f"\nTransformation vector similarities:")
    print(f"(If high, the capital-of relationship is geometrically consistent)")
    
    similarities = []
    for i, t1 in enumerate(transformations):
        for j, t2 in enumerate(transformations):
            if i < j:
                cos_sim = torch.nn.functional.cosine_similarity(
                    t1['transform'].unsqueeze(0),
                    t2['transform'].unsqueeze(0)
                ).item()
                similarities.append(cos_sim)
                print(f"  {t1['country']}→{t1['capital']} vs {t2['country']}→{t2['capital']}: {cos_sim:.4f}")
    
    print(f"\nMean similarity: {np.mean(similarities):.4f}")
    print(f"Std similarity:  {np.std(similarities):.4f}")
    
    # Compute mean transformation
    mean_transform = torch.stack([t['transform'] for t in transformations]).mean(dim=0)
    
    print(f"\n--- Testing Mean Transformation ---")
    print(f"Can we predict capitals using the mean transformation?")
    
    correct = 0
    total = 0
    
    for t in transformations:
        country_ids = tokenizer.encode(t['country'], add_special_tokens=False)
        country_emb = embeddings[country_ids[0]]
        
        # Predict capital using mean transformation
        predicted_emb = country_emb + mean_transform
        
        # Find nearest token
        distances = (embeddings - predicted_emb.unsqueeze(0)).norm(dim=1)
        nearest_idx = distances.argmin().item()
        nearest_token = tokenizer.decode([nearest_idx])
        
        is_correct = nearest_token.strip().lower() == t['capital'].lower()
        if is_correct:
            correct += 1
        total += 1
        
        print(f"  {t['country']} + mean_transform → {repr(nearest_token)} (expected: {t['capital']}) {'✓' if is_correct else '✗'}")
    
    print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
    
    return transformations, mean_transform


def analyze_sign_patterns(embeddings, tokenizer):
    """
    Analyze sign patterns of content tokens.
    
    From Doc 164: Signs encode semantics.
    Do countries share sign patterns? Do capitals?
    """
    print("\n" + "=" * 70)
    print("SIGN PATTERN ANALYSIS")
    print("=" * 70)
    
    countries = ["France", "Germany", "Italy", "Spain", "Japan", "China"]
    capitals = ["Paris", "Berlin", "Rome", "Madrid", "Tokyo", "Beijing"]
    
    country_signs = []
    capital_signs = []
    
    for country in countries:
        ids = tokenizer.encode(country, add_special_tokens=False)
        if ids:
            signs = torch.sign(embeddings[ids[0]])
            country_signs.append(signs)
    
    for capital in capitals:
        ids = tokenizer.encode(capital, add_special_tokens=False)
        if ids:
            signs = torch.sign(embeddings[ids[0]])
            capital_signs.append(signs)
    
    if country_signs and capital_signs:
        country_signs = torch.stack(country_signs)
        capital_signs = torch.stack(capital_signs)
        
        # Sign agreement within countries
        country_agreement = (country_signs[0] == country_signs).float().mean(dim=1)
        print(f"\nSign agreement with France (within countries):")
        for i, country in enumerate(countries):
            print(f"  {country}: {country_agreement[i].item()*100:.1f}%")
        
        # Sign agreement within capitals
        capital_agreement = (capital_signs[0] == capital_signs).float().mean(dim=1)
        print(f"\nSign agreement with Paris (within capitals):")
        for i, capital in enumerate(capitals):
            print(f"  {capital}: {capital_agreement[i].item()*100:.1f}%")
        
        # Sign difference between country and capital
        print(f"\nSign difference (country → capital):")
        for i, (country, capital) in enumerate(zip(countries, capitals)):
            diff = (country_signs[i] != capital_signs[i]).float().mean().item()
            print(f"  {country} → {capital}: {diff*100:.1f}% signs differ")
        
        # Find which dimensions consistently flip
        all_flips = (country_signs != capital_signs).float().mean(dim=0)
        consistent_flip_dims = (all_flips > 0.8).nonzero().squeeze()
        
        print(f"\nDimensions that consistently flip (>80% of pairs):")
        print(f"  Count: {len(consistent_flip_dims)}")
        if len(consistent_flip_dims) > 0 and len(consistent_flip_dims) < 20:
            print(f"  Dimensions: {consistent_flip_dims.tolist()}")


def main():
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    
    embeddings = model.model.embed_tokens.weight.data.float().cpu()
    
    # Run analyses
    embeddings_cpu, levels, signs = analyze_embedding_structure(model, tokenizer)
    
    find_zero_axis_tokens(embeddings, tokenizer)
    
    analyze_content_tokens(embeddings, tokenizer)
    
    transformations, mean_transform = analyze_capital_relationship(embeddings, tokenizer)
    
    analyze_sign_patterns(embeddings, tokenizer)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key questions answered:
1. Do content tokens cluster? → Check sign pattern analysis
2. Is capital-of geometric? → Check transformation similarity
3. Can we predict capitals? → Check mean transformation accuracy
""")


if __name__ == "__main__":
    main()
