#!/usr/bin/env python3
"""
φ-Lattice Navigation: Using Rules to Make Discoveries
======================================================

We know the rules of the game board. Now let's use them to:
1. PREDICT where solutions should exist
2. NAVIGATE to those positions
3. VERIFY if our predictions are correct

Navigation Experiments:
1. Find opposites via sign flipping
2. Find intermediates via interpolation
3. Predict gender counterparts via level shift
4. Find related concepts via φ-harmonics
5. Discover NEW relationships by exploring the lattice
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """Get embedding for a single word."""
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        return None
    return model.model.embed_tokens.weight[ids[0]].detach()


def find_nearest_tokens(model, tokenizer, target_embed, top_k=10, exclude_ids=None):
    """Find tokens nearest to target embedding."""
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
# NAVIGATION 1: Find Opposites via Sign Flipping
# =============================================================================

def navigate_opposites(model, tokenizer):
    """
    Rule: Sign flipping = conceptual transformation
    Prediction: 100% sign flip should give conceptual opposite
    """
    print("="*70)
    print("NAVIGATION 1: FINDING OPPOSITES VIA SIGN FLIP")
    print("="*70)
    
    test_words = ["good", "happy", "light", "hot", "fast", "big", "love", "peace"]
    
    print("\nPrediction: 100% sign flip → conceptual opposite")
    print("-"*70)
    
    for word in test_words:
        embed = get_token_embedding(model, tokenizer, word)
        if embed is None:
            continue
        
        # Encode to φ-lattice
        levels, signs = encode_phi(embed)
        
        # Flip ALL signs
        signs_flipped = -signs
        
        # Decode back
        opposite_embed = decode_phi(levels, signs_flipped).to(embed.dtype)
        
        # Find nearest tokens
        word_id = tokenizer.encode(word, add_special_tokens=False)[0]
        nearest = find_nearest_tokens(model, tokenizer, opposite_embed, top_k=5, exclude_ids=[word_id])
        
        print(f"\n{word:10s} → sign flip → ", end="")
        for token, sim, _ in nearest[:3]:
            print(f"'{token.strip()}' ({sim:.3f}), ", end="")
        print()


# =============================================================================
# NAVIGATION 2: Find Intermediates via Interpolation
# =============================================================================

def navigate_interpolation(model, tokenizer):
    """
    Rule: Interpolation preserves coherence
    Prediction: Midpoint between opposites should give intermediate concept
    """
    print("\n" + "="*70)
    print("NAVIGATION 2: FINDING INTERMEDIATES VIA INTERPOLATION")
    print("="*70)
    
    pairs = [
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("young", "old"),
        ("rich", "poor"),
        ("happy", "sad"),
    ]
    
    print("\nPrediction: Midpoint should give intermediate concept")
    print("-"*70)
    
    for word1, word2 in pairs:
        embed1 = get_token_embedding(model, tokenizer, word1)
        embed2 = get_token_embedding(model, tokenizer, word2)
        
        if embed1 is None or embed2 is None:
            continue
        
        # Encode to φ-lattice
        levels1, signs1 = encode_phi(embed1)
        levels2, signs2 = encode_phi(embed2)
        
        # Interpolate at midpoint (t=0.5)
        levels_mid = ((levels1.float() + levels2.float()) / 2).round().to(torch.int16)
        # For signs, use element-wise: if same, keep; if different, random
        signs_mid = torch.where(signs1 == signs2, signs1, 
                                torch.where(torch.rand_like(signs1.float()) > 0.5, signs1, signs2))
        
        mid_embed = decode_phi(levels_mid, signs_mid).to(embed1.dtype)
        
        # Find nearest tokens
        id1 = tokenizer.encode(word1, add_special_tokens=False)[0]
        id2 = tokenizer.encode(word2, add_special_tokens=False)[0]
        nearest = find_nearest_tokens(model, tokenizer, mid_embed, top_k=5, exclude_ids=[id1, id2])
        
        print(f"\n{word1:6s} ↔ {word2:6s} midpoint → ", end="")
        for token, sim, _ in nearest[:3]:
            print(f"'{token.strip()}' ({sim:.3f}), ", end="")
        print()


# =============================================================================
# NAVIGATION 3: Predict Gender Counterpart via Level Shift
# =============================================================================

def navigate_gender(model, tokenizer):
    """
    Rule: Gender direction is ~-10 levels (female lower than male)
    Prediction: Shifting levels by -10 should move toward female counterpart
    """
    print("\n" + "="*70)
    print("NAVIGATION 3: PREDICTING GENDER COUNTERPARTS")
    print("="*70)
    
    # Test: shift male words by -10 levels, see if we get female
    male_words = ["king", "man", "boy", "father", "brother", "uncle", "prince", "actor"]
    expected_female = ["queen", "woman", "girl", "mother", "sister", "aunt", "princess", "actress"]
    
    print("\nPrediction: male - 10 levels → female counterpart")
    print("-"*70)
    
    for male, female in zip(male_words, expected_female):
        embed_male = get_token_embedding(model, tokenizer, male)
        embed_female = get_token_embedding(model, tokenizer, female)
        
        if embed_male is None:
            continue
        
        # Encode to φ-lattice
        levels, signs = encode_phi(embed_male)
        
        # Shift levels by -10 (toward female)
        levels_shifted = levels - 10
        
        # Decode back
        predicted_embed = decode_phi(levels_shifted, signs).to(embed_male.dtype)
        
        # Find nearest tokens
        male_id = tokenizer.encode(male, add_special_tokens=False)[0]
        nearest = find_nearest_tokens(model, tokenizer, predicted_embed, top_k=5, exclude_ids=[male_id])
        
        # Check if expected female is in top results
        found_expected = any(female.lower() in token.lower() for token, _, _ in nearest[:5])
        marker = "✓" if found_expected else "✗"
        
        print(f"\n{male:10s} - 10 levels → ", end="")
        for token, sim, _ in nearest[:3]:
            print(f"'{token.strip()}' ({sim:.3f}), ", end="")
        print(f"  [expected: {female}] {marker}")


# =============================================================================
# NAVIGATION 4: Find Related Concepts via φ-Harmonics
# =============================================================================

def navigate_harmonics(model, tokenizer):
    """
    Rule: Clustering at Δ=64 (φ^0.5)
    Prediction: Tokens at ±64 levels should be semantically related
    """
    print("\n" + "="*70)
    print("NAVIGATION 4: FINDING RELATED CONCEPTS VIA φ-HARMONICS")
    print("="*70)
    
    test_words = ["science", "music", "food", "money", "love", "war"]
    
    print("\nPrediction: ±64 levels (φ^0.5 harmonic) → related concepts")
    print("-"*70)
    
    for word in test_words:
        embed = get_token_embedding(model, tokenizer, word)
        if embed is None:
            continue
        
        levels, signs = encode_phi(embed)
        word_id = tokenizer.encode(word, add_special_tokens=False)[0]
        
        # Shift by +64 (φ^0.5 harmonic)
        levels_up = levels + 64
        embed_up = decode_phi(levels_up, signs).to(embed.dtype)
        nearest_up = find_nearest_tokens(model, tokenizer, embed_up, top_k=3, exclude_ids=[word_id])
        
        # Shift by -64
        levels_down = levels - 64
        embed_down = decode_phi(levels_down, signs).to(embed.dtype)
        nearest_down = find_nearest_tokens(model, tokenizer, embed_down, top_k=3, exclude_ids=[word_id])
        
        print(f"\n{word:10s}:")
        print(f"  +64 levels → ", end="")
        for token, sim, _ in nearest_up:
            print(f"'{token.strip()}' ({sim:.3f}), ", end="")
        print()
        print(f"  -64 levels → ", end="")
        for token, sim, _ in nearest_down:
            print(f"'{token.strip()}' ({sim:.3f}), ", end="")
        print()


# =============================================================================
# NAVIGATION 5: Discover NEW Relationships
# =============================================================================

def navigate_discovery(model, tokenizer):
    """
    Use rules to discover relationships we didn't know about.
    """
    print("\n" + "="*70)
    print("NAVIGATION 5: DISCOVERING NEW RELATIONSHIPS")
    print("="*70)
    
    # Discovery 1: What's the "opposite" of abstract concepts?
    print("\n--- Discovery 1: Opposites of Abstract Concepts ---")
    abstract_words = ["truth", "beauty", "justice", "freedom", "wisdom"]
    
    for word in abstract_words:
        embed = get_token_embedding(model, tokenizer, word)
        if embed is None:
            continue
        
        levels, signs = encode_phi(embed)
        signs_flipped = -signs
        opposite_embed = decode_phi(levels, signs_flipped).to(embed.dtype)
        
        word_id = tokenizer.encode(word, add_special_tokens=False)[0]
        nearest = find_nearest_tokens(model, tokenizer, opposite_embed, top_k=3, exclude_ids=[word_id])
        
        print(f"{word:10s} → opposite → ", end="")
        for token, sim, _ in nearest:
            print(f"'{token.strip()}' ({sim:.3f}), ", end="")
        print()
    
    # Discovery 2: What's at extreme level shifts?
    print("\n--- Discovery 2: Extreme Level Shifts ---")
    base_word = "human"
    embed = get_token_embedding(model, tokenizer, base_word)
    levels, signs = encode_phi(embed)
    word_id = tokenizer.encode(base_word, add_special_tokens=False)[0]
    
    for shift in [-200, -100, -50, 0, +50, +100, +200]:
        levels_shifted = levels + shift
        shifted_embed = decode_phi(levels_shifted, signs).to(embed.dtype)
        nearest = find_nearest_tokens(model, tokenizer, shifted_embed, top_k=3, exclude_ids=[word_id])
        
        print(f"human {shift:+4d} levels → ", end="")
        for token, sim, _ in nearest:
            print(f"'{token.strip()}' ({sim:.3f}), ", end="")
        print()
    
    # Discovery 3: Combining concepts
    print("\n--- Discovery 3: Concept Combinations ---")
    combinations = [
        ("fire", "water"),
        ("science", "art"),
        ("past", "future"),
        ("mind", "body"),
    ]
    
    for word1, word2 in combinations:
        embed1 = get_token_embedding(model, tokenizer, word1)
        embed2 = get_token_embedding(model, tokenizer, word2)
        
        if embed1 is None or embed2 is None:
            continue
        
        levels1, signs1 = encode_phi(embed1)
        levels2, signs2 = encode_phi(embed2)
        
        # Combine: average levels, multiply signs
        levels_combined = ((levels1.float() + levels2.float()) / 2).round().to(torch.int16)
        signs_combined = signs1 * signs2
        
        combined_embed = decode_phi(levels_combined, signs_combined).to(embed1.dtype)
        
        id1 = tokenizer.encode(word1, add_special_tokens=False)[0]
        id2 = tokenizer.encode(word2, add_special_tokens=False)[0]
        nearest = find_nearest_tokens(model, tokenizer, combined_embed, top_k=3, exclude_ids=[id1, id2])
        
        print(f"{word1:8s} × {word2:8s} → ", end="")
        for token, sim, _ in nearest:
            print(f"'{token.strip()}' ({sim:.3f}), ", end="")
        print()


def main():
    print("="*70)
    print("φ-LATTICE NAVIGATION: USING RULES TO MAKE DISCOVERIES")
    print("="*70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    navigate_opposites(model, tokenizer)
    navigate_interpolation(model, tokenizer)
    navigate_gender(model, tokenizer)
    navigate_harmonics(model, tokenizer)
    navigate_discovery(model, tokenizer)
    
    print("\n" + "="*70)
    print("NAVIGATION COMPLETE")
    print("="*70)
    print("""
SUMMARY:
- Sign flipping finds conceptual transformations (not always "opposites")
- Interpolation finds intermediate concepts
- Level shifts navigate semantic dimensions (gender, scale, etc.)
- φ-harmonics (±64) find related concepts
- Combinations create novel concept blends

The φ-lattice is a NAVIGABLE space. We can use the rules to:
1. PREDICT where concepts should be
2. NAVIGATE to those positions
3. DISCOVER what's actually there
""")


if __name__ == "__main__":
    main()
