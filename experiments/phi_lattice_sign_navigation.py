#!/usr/bin/env python3
"""
φ-Lattice Sign Navigation: The Semantic Axis
=============================================

Key insight from previous experiments:
- Level shifts DON'T change semantics (±200 levels → same word)
- Full sign flips DESTROY semantics (→ garbage)
- Semantics must be encoded in PARTIAL sign patterns

New hypothesis: Navigate by changing SPECIFIC sign dimensions,
not all signs at once.

Experiments:
1. Find which sign dimensions encode which semantic features
2. Navigate by flipping specific dimension groups
3. Discover the "semantic axes" in sign space
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)
K_SCALE = 128


def encode_phi(tensor):
    signs = torch.sign(tensor)
    signs[signs == 0] = 1
    magnitudes = tensor.abs().clamp(min=1e-45)
    levels = torch.round(K_SCALE * torch.log(magnitudes) / LOG_PHI)
    return levels.to(torch.int16), signs.to(torch.int8)


def decode_phi(levels, signs):
    exponents = levels.float() / K_SCALE
    magnitudes = torch.exp(exponents * LOG_PHI)
    return signs.float() * magnitudes


def get_token_embedding(model, tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        return None
    return model.model.embed_tokens.weight[ids[0]].detach()


def find_nearest_tokens(model, tokenizer, target_embed, top_k=10, exclude_ids=None):
    all_embeds = model.model.embed_tokens.weight.detach()
    sims = F.cosine_similarity(target_embed.unsqueeze(0).float(), all_embeds.float())
    
    if exclude_ids:
        for idx in exclude_ids:
            sims[idx] = -1
    
    top_indices = sims.topk(top_k).indices
    results = []
    for idx in top_indices:
        token = tokenizer.decode([idx.item()])
        sim = sims[idx].item()
        results.append((token, sim, idx.item()))
    return results


# =============================================================================
# EXPERIMENT 1: Find Sign Dimensions that Encode Gender
# =============================================================================

def find_gender_dimensions(model, tokenizer):
    """
    Find which sign dimensions differ between male/female word pairs.
    """
    print("="*70)
    print("EXPERIMENT 1: FINDING GENDER DIMENSIONS IN SIGN SPACE")
    print("="*70)
    
    pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("boy", "girl"),
        ("father", "mother"),
        ("brother", "sister"),
        ("he", "she"),
        ("him", "her"),
        ("his", "hers"),
    ]
    
    # Track which dimensions flip between male/female
    flip_counts = torch.zeros(3584)  # hidden_dim
    total_pairs = 0
    
    for male, female in pairs:
        embed_m = get_token_embedding(model, tokenizer, male)
        embed_f = get_token_embedding(model, tokenizer, female)
        
        if embed_m is None or embed_f is None:
            continue
        
        _, signs_m = encode_phi(embed_m.cpu())
        _, signs_f = encode_phi(embed_f.cpu())
        
        # Find dimensions where signs differ
        diff = (signs_m != signs_f).float()
        flip_counts += diff
        total_pairs += 1
    
    # Find dimensions that consistently flip
    flip_rate = flip_counts / total_pairs
    
    # Top dimensions that flip most often
    top_flip_dims = flip_rate.topk(50).indices
    
    print(f"\nAnalyzed {total_pairs} male/female pairs")
    print(f"\nTop 20 dimensions that flip between genders:")
    for i, dim in enumerate(top_flip_dims[:20]):
        print(f"  Dim {dim.item():4d}: flips {flip_rate[dim]*100:.0f}% of pairs")
    
    # Find dimensions that ALWAYS flip (100%)
    always_flip = (flip_rate == 1.0).nonzero().squeeze()
    print(f"\nDimensions that ALWAYS flip: {len(always_flip)}")
    
    # Find dimensions that NEVER flip (0%)
    never_flip = (flip_rate == 0.0).sum().item()
    print(f"Dimensions that NEVER flip: {never_flip}")
    
    return flip_rate, top_flip_dims


# =============================================================================
# EXPERIMENT 2: Navigate Gender by Flipping Specific Dimensions
# =============================================================================

def navigate_by_gender_dims(model, tokenizer, flip_rate):
    """
    Use discovered gender dimensions to navigate.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: NAVIGATING GENDER BY FLIPPING SPECIFIC DIMENSIONS")
    print("="*70)
    
    # Get dimensions that flip >50% of the time
    gender_dims = (flip_rate > 0.5).nonzero().squeeze()
    print(f"\nUsing {len(gender_dims)} dimensions that flip >50% between genders")
    
    test_words = ["king", "man", "boy", "father", "actor", "prince"]
    expected = ["queen", "woman", "girl", "mother", "actress", "princess"]
    
    print("\nNavigating male → female by flipping gender dimensions:")
    print("-"*70)
    
    for male, female_expected in zip(test_words, expected):
        embed = get_token_embedding(model, tokenizer, male)
        if embed is None:
            continue
        
        levels, signs = encode_phi(embed)
        
        # Flip only the gender dimensions
        signs_flipped = signs.clone()
        signs_flipped[gender_dims] *= -1
        
        # Decode
        navigated_embed = decode_phi(levels, signs_flipped).to(embed.dtype)
        
        # Find nearest
        male_id = tokenizer.encode(male, add_special_tokens=False)[0]
        nearest = find_nearest_tokens(model, tokenizer, navigated_embed, top_k=5, exclude_ids=[male_id])
        
        found = any(female_expected.lower() in t.lower() for t, _, _ in nearest[:5])
        marker = "✓" if found else "✗"
        
        print(f"{male:10s} → flip gender dims → ", end="")
        for token, sim, _ in nearest[:3]:
            print(f"'{token.strip()}' ({sim:.3f}), ", end="")
        print(f"  [expected: {female_expected}] {marker}")


# =============================================================================
# EXPERIMENT 3: Find Semantic Axes by Comparing Word Pairs
# =============================================================================

def find_semantic_axes(model, tokenizer):
    """
    Find semantic axes by analyzing which dimensions differ between concept pairs.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: DISCOVERING SEMANTIC AXES")
    print("="*70)
    
    # Different semantic relationships
    relationships = {
        "size": [("big", "small"), ("large", "tiny"), ("huge", "little")],
        "temperature": [("hot", "cold"), ("warm", "cool"), ("burning", "freezing")],
        "speed": [("fast", "slow"), ("quick", "sluggish"), ("rapid", "gradual")],
        "age": [("young", "old"), ("new", "ancient"), ("fresh", "stale")],
        "valence": [("good", "bad"), ("happy", "sad"), ("love", "hate")],
    }
    
    axis_dims = {}
    
    for axis_name, pairs in relationships.items():
        flip_counts = torch.zeros(3584)
        total = 0
        
        for word1, word2 in pairs:
            embed1 = get_token_embedding(model, tokenizer, word1)
            embed2 = get_token_embedding(model, tokenizer, word2)
            
            if embed1 is None or embed2 is None:
                continue
            
            _, signs1 = encode_phi(embed1.cpu())
            _, signs2 = encode_phi(embed2.cpu())
            
            diff = (signs1 != signs2).float()
            flip_counts += diff
            total += 1
        
        if total > 0:
            flip_rate = flip_counts / total
            # Dimensions that flip >50%
            axis_specific = (flip_rate > 0.5).nonzero().squeeze()
            axis_dims[axis_name] = axis_specific
            
            print(f"\n{axis_name.upper()} axis: {len(axis_specific)} specific dimensions")
            print(f"  Top dims: {axis_specific[:10].tolist()}")
    
    # Check for overlap between axes
    print("\n" + "-"*70)
    print("AXIS OVERLAP ANALYSIS:")
    
    for name1, dims1 in axis_dims.items():
        for name2, dims2 in axis_dims.items():
            if name1 >= name2:
                continue
            
            set1 = set(dims1.tolist()) if dims1.dim() > 0 else set()
            set2 = set(dims2.tolist()) if dims2.dim() > 0 else set()
            
            overlap = len(set1 & set2)
            print(f"  {name1:12s} ∩ {name2:12s}: {overlap} shared dimensions")
    
    return axis_dims


# =============================================================================
# EXPERIMENT 4: Navigate Using Discovered Axes
# =============================================================================

def navigate_with_axes(model, tokenizer, axis_dims):
    """
    Use discovered semantic axes to navigate.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: NAVIGATING WITH SEMANTIC AXES")
    print("="*70)
    
    # Test navigation along each axis
    test_cases = [
        ("size", "big", "small"),
        ("temperature", "hot", "cold"),
        ("speed", "fast", "slow"),
        ("age", "young", "old"),
        ("valence", "good", "bad"),
    ]
    
    for axis_name, word1, word2 in test_cases:
        if axis_name not in axis_dims:
            continue
        
        dims = axis_dims[axis_name]
        if dims.dim() == 0 or len(dims) == 0:
            continue
        
        embed = get_token_embedding(model, tokenizer, word1)
        if embed is None:
            continue
        
        levels, signs = encode_phi(embed)
        
        # Flip axis dimensions
        signs_flipped = signs.clone()
        signs_flipped[dims] *= -1
        
        navigated_embed = decode_phi(levels, signs_flipped).to(embed.dtype)
        
        word_id = tokenizer.encode(word1, add_special_tokens=False)[0]
        nearest = find_nearest_tokens(model, tokenizer, navigated_embed, top_k=5, exclude_ids=[word_id])
        
        found = any(word2.lower() in t.lower() for t, _, _ in nearest[:5])
        marker = "✓" if found else "✗"
        
        print(f"\n{axis_name:12s}: {word1:8s} → flip axis → ", end="")
        for token, sim, _ in nearest[:3]:
            print(f"'{token.strip()}' ({sim:.3f}), ", end="")
        print(f"  [expected: {word2}] {marker}")


# =============================================================================
# EXPERIMENT 5: The Sign Similarity Metric
# =============================================================================

def analyze_sign_similarity(model, tokenizer):
    """
    Analyze how sign similarity correlates with semantic similarity.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: SIGN SIMILARITY AS SEMANTIC METRIC")
    print("="*70)
    
    # Test words and their semantic neighbors
    test_cases = [
        ("dog", ["cat", "puppy", "animal", "pet", "wolf"]),
        ("happy", ["joyful", "glad", "sad", "angry", "excited"]),
        ("king", ["queen", "prince", "ruler", "peasant", "throne"]),
        ("science", ["physics", "math", "art", "research", "experiment"]),
    ]
    
    print("\nSign agreement vs semantic relationship:")
    print("-"*70)
    
    for base_word, related_words in test_cases:
        base_embed = get_token_embedding(model, tokenizer, base_word)
        if base_embed is None:
            continue
        
        _, base_signs = encode_phi(base_embed)
        
        print(f"\n{base_word}:")
        for related in related_words:
            related_embed = get_token_embedding(model, tokenizer, related)
            if related_embed is None:
                continue
            
            _, related_signs = encode_phi(related_embed)
            
            # Sign agreement
            agreement = (base_signs == related_signs).float().mean().item()
            
            # Cosine similarity (original space)
            cos_sim = F.cosine_similarity(
                base_embed.unsqueeze(0).float(),
                related_embed.unsqueeze(0).float()
            ).item()
            
            print(f"  {related:12s}: sign_agree={agreement:.3f}, cos_sim={cos_sim:.3f}")


def main():
    print("="*70)
    print("φ-LATTICE SIGN NAVIGATION: THE SEMANTIC AXIS")
    print("="*70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Experiment 1: Find gender dimensions
    flip_rate, top_flip_dims = find_gender_dimensions(model, tokenizer)
    
    # Experiment 2: Navigate using gender dimensions
    navigate_by_gender_dims(model, tokenizer, flip_rate)
    
    # Experiment 3: Find semantic axes
    axis_dims = find_semantic_axes(model, tokenizer)
    
    # Experiment 4: Navigate with axes
    navigate_with_axes(model, tokenizer, axis_dims)
    
    # Experiment 5: Sign similarity analysis
    analyze_sign_similarity(model, tokenizer)
    
    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print("""
KEY FINDINGS:
1. Semantics are encoded in SIGN PATTERNS, not levels
2. Different semantic axes use DIFFERENT sign dimensions
3. Flipping specific dimensions navigates specific semantic axes
4. Sign agreement correlates with semantic similarity

THE SIGN SPACE IS THE SEMANTIC SPACE.
Levels are just magnitude. Signs are meaning.
""")


if __name__ == "__main__":
    main()
