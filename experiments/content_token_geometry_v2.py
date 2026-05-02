#!/usr/bin/env python3
"""
Content Token Geometry v2: Investigating the 295 Flip Dimensions
=================================================================

Key finding from v1: 295 dimensions consistently flip when going
from country → capital.

Hypothesis: These 295 dimensions encode the "capital-of" relationship.
If we flip ONLY these dimensions, can we navigate from country to capital?

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


def find_capital_flip_dimensions(embeddings, tokenizer):
    """Find dimensions that consistently flip for country→capital."""
    
    print("=" * 70)
    print("FINDING CAPITAL-OF FLIP DIMENSIONS")
    print("=" * 70)
    
    capitals = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
        ("Poland", "Warsaw"),
        ("Japan", "Tokyo"),
        ("China", "Beijing"),
        ("Egypt", "Cairo"),
        ("Greece", "Athens"),
        ("Sweden", "Stockholm"),
        ("Norway", "Oslo"),
        ("Denmark", "Copenhagen"),
        ("Austria", "Vienna"),
        ("Portugal", "Lisbon"),
        ("Netherlands", "Amsterdam"),
    ]
    
    flip_counts = torch.zeros(embeddings.shape[1])
    valid_pairs = 0
    
    for country, capital in capitals:
        country_ids = tokenizer.encode(country, add_special_tokens=False)
        capital_ids = tokenizer.encode(capital, add_special_tokens=False)
        
        if country_ids and capital_ids:
            country_emb = embeddings[country_ids[0]]
            capital_emb = embeddings[capital_ids[0]]
            
            # Find dimensions where sign flips
            country_signs = torch.sign(country_emb)
            capital_signs = torch.sign(capital_emb)
            
            flips = (country_signs != capital_signs).float()
            flip_counts += flips
            valid_pairs += 1
    
    # Find dimensions that flip in >80% of pairs
    flip_rate = flip_counts / valid_pairs
    consistent_flip_dims = (flip_rate > 0.8).nonzero().squeeze().tolist()
    
    if isinstance(consistent_flip_dims, int):
        consistent_flip_dims = [consistent_flip_dims]
    
    print(f"\nAnalyzed {valid_pairs} country-capital pairs")
    print(f"Dimensions that flip in >80% of pairs: {len(consistent_flip_dims)}")
    
    # Also find dimensions that NEVER flip (stable)
    stable_dims = (flip_rate < 0.2).nonzero().squeeze().tolist()
    if isinstance(stable_dims, int):
        stable_dims = [stable_dims]
    
    print(f"Dimensions that rarely flip (<20%): {len(stable_dims)}")
    
    return consistent_flip_dims, stable_dims, flip_rate


def test_sign_flip_navigation(embeddings, tokenizer, flip_dims):
    """
    Test: Can we navigate from country to capital by flipping signs?
    
    If the 295 dimensions encode "capital-of", then:
    country_emb with flipped signs in those dims → should be near capital_emb
    """
    print("\n" + "=" * 70)
    print("TESTING SIGN FLIP NAVIGATION")
    print("=" * 70)
    
    test_pairs = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Japan", "Tokyo"),
        ("Brazil", "Brasilia"),
        ("Mexico", "Mexico"),  # Tricky - same name!
        ("India", "Delhi"),
        ("Russia", "Moscow"),
    ]
    
    flip_mask = torch.zeros(embeddings.shape[1], dtype=torch.bool)
    flip_mask[flip_dims] = True
    
    correct = 0
    total = 0
    
    for country, expected_capital in test_pairs:
        country_ids = tokenizer.encode(country, add_special_tokens=False)
        
        if not country_ids:
            continue
        
        country_emb = embeddings[country_ids[0]].clone()
        
        # Flip signs in the capital-of dimensions
        flipped_emb = country_emb.clone()
        flipped_emb[flip_mask] = -flipped_emb[flip_mask]
        
        # Find nearest token
        distances = (embeddings - flipped_emb.unsqueeze(0)).norm(dim=1)
        nearest_idx = distances.argmin().item()
        nearest_token = tokenizer.decode([nearest_idx]).strip()
        
        # Also find top-5
        top5_indices = distances.argsort()[:5]
        top5_tokens = [tokenizer.decode([idx.item()]).strip() for idx in top5_indices]
        
        is_correct = nearest_token.lower() == expected_capital.lower()
        in_top5 = any(t.lower() == expected_capital.lower() for t in top5_tokens)
        
        if is_correct:
            correct += 1
        total += 1
        
        status = "✓" if is_correct else ("(top5)" if in_top5 else "✗")
        print(f"\n{country} → flip signs → {nearest_token} (expected: {expected_capital}) {status}")
        print(f"  Top 5: {top5_tokens}")
    
    print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
    
    return correct / total


def test_magnitude_preservation(embeddings, tokenizer, flip_dims):
    """
    Test: Do we need to adjust magnitudes, or just signs?
    
    From Doc 164: Levels encode energy, Signs encode semantics.
    If we preserve levels and only flip signs, do we get better results?
    """
    print("\n" + "=" * 70)
    print("TESTING MAGNITUDE-PRESERVING NAVIGATION")
    print("=" * 70)
    
    test_pairs = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Japan", "Tokyo"),
    ]
    
    for country, expected_capital in test_pairs:
        country_ids = tokenizer.encode(country, add_special_tokens=False)
        capital_ids = tokenizer.encode(expected_capital, add_special_tokens=False)
        
        if not country_ids or not capital_ids:
            continue
        
        country_emb = embeddings[country_ids[0]]
        capital_emb = embeddings[capital_ids[0]]
        
        # Method 1: Just flip signs (preserve country magnitudes)
        flipped_emb = country_emb.clone()
        flip_mask = torch.zeros(embeddings.shape[1], dtype=torch.bool)
        flip_mask[flip_dims] = True
        flipped_emb[flip_mask] = -flipped_emb[flip_mask]
        
        dist_flip = (capital_emb - flipped_emb).norm().item()
        
        # Method 2: Use capital's magnitudes with country's signs (flipped)
        hybrid_emb = capital_emb.abs() * torch.sign(flipped_emb)
        dist_hybrid = (capital_emb - hybrid_emb).norm().item()
        
        # Method 3: Direct distance
        dist_direct = (capital_emb - country_emb).norm().item()
        
        print(f"\n{country} → {expected_capital}:")
        print(f"  Direct distance:        {dist_direct:.4f}")
        print(f"  After sign flip:        {dist_flip:.4f}")
        print(f"  Hybrid (capital mags):  {dist_hybrid:.4f}")


def analyze_flip_dimension_semantics(embeddings, tokenizer, flip_dims, stable_dims):
    """
    What do the flip dimensions encode?
    
    Look at which tokens have extreme values in these dimensions.
    """
    print("\n" + "=" * 70)
    print("FLIP DIMENSION SEMANTICS")
    print("=" * 70)
    
    # For each flip dimension, find tokens with extreme values
    print(f"\nAnalyzing {len(flip_dims)} flip dimensions...")
    
    # Sample a few flip dimensions
    sample_dims = flip_dims[:5] if len(flip_dims) > 5 else flip_dims
    
    for dim in sample_dims:
        values = embeddings[:, dim]
        
        # Top positive
        top_pos_idx = values.argsort(descending=True)[:5]
        top_pos_tokens = [tokenizer.decode([idx.item()]) for idx in top_pos_idx]
        
        # Top negative
        top_neg_idx = values.argsort()[:5]
        top_neg_tokens = [tokenizer.decode([idx.item()]) for idx in top_neg_idx]
        
        print(f"\nDimension {dim}:")
        print(f"  Most positive: {[repr(t) for t in top_pos_tokens]}")
        print(f"  Most negative: {[repr(t) for t in top_neg_tokens]}")


def test_relationship_specific_dims(embeddings, tokenizer):
    """
    Test: Do different relationships have different flip dimensions?
    
    Compare:
    - country → capital
    - person → profession
    - animal → sound
    """
    print("\n" + "=" * 70)
    print("RELATIONSHIP-SPECIFIC DIMENSIONS")
    print("=" * 70)
    
    relationships = {
        "capital-of": [
            ("France", "Paris"),
            ("Germany", "Berlin"),
            ("Italy", "Rome"),
            ("Japan", "Tokyo"),
        ],
        "profession": [
            ("Einstein", "physicist"),
            ("Shakespeare", "playwright"),
            ("Mozart", "composer"),
            ("Picasso", "painter"),
        ],
    }
    
    for rel_name, pairs in relationships.items():
        flip_counts = torch.zeros(embeddings.shape[1])
        valid = 0
        
        for entity, related in pairs:
            entity_ids = tokenizer.encode(entity, add_special_tokens=False)
            related_ids = tokenizer.encode(related, add_special_tokens=False)
            
            if entity_ids and related_ids:
                entity_signs = torch.sign(embeddings[entity_ids[0]])
                related_signs = torch.sign(embeddings[related_ids[0]])
                
                flips = (entity_signs != related_signs).float()
                flip_counts += flips
                valid += 1
        
        if valid > 0:
            flip_rate = flip_counts / valid
            high_flip_dims = (flip_rate > 0.75).sum().item()
            
            print(f"\n{rel_name}:")
            print(f"  Pairs analyzed: {valid}")
            print(f"  Dimensions with >75% flip rate: {high_flip_dims}")


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
    
    # Find flip dimensions
    flip_dims, stable_dims, flip_rate = find_capital_flip_dimensions(embeddings, tokenizer)
    
    # Test sign flip navigation
    test_sign_flip_navigation(embeddings, tokenizer, flip_dims)
    
    # Test magnitude preservation
    test_magnitude_preservation(embeddings, tokenizer, flip_dims)
    
    # Analyze what flip dimensions encode
    analyze_flip_dimension_semantics(embeddings, tokenizer, flip_dims, stable_dims)
    
    # Test relationship-specific dimensions
    test_relationship_specific_dims(embeddings, tokenizer)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
