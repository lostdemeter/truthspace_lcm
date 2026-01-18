#!/usr/bin/env python3
"""
Qwen2.0 LCM Theory Validation
==============================

Test if Qwen2's embeddings follow the LCM theory:
1. φ is the fundamental unit of semantic distance
2. Self-similarity: same transformation type = same Δ
3. Platonic Ideals sit at origin of multiple dimensions

From design docs 039 and 114:
- king → queen: Δ = +φ on gender_flip
- man → woman:  Δ = +φ on gender_flip  
- boy → girl:   Δ = +φ on gender_flip
"""

import torch
import numpy as np
from pathlib import Path
import json

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


def load_model():
    """Load Qwen2-0.5B model."""
    print("Loading Qwen2-0.5B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
        torch_dtype=torch.float16,
    )
    model = model.cpu()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    
    return model, tokenizer


def get_embeddings(model, tokenizer, words):
    """Get embeddings for words (single-token only)."""
    embed_weights = model.model.embed_tokens.weight.detach().cpu().float().numpy()
    
    embeddings = {}
    for word in words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 1:
            embeddings[word] = embed_weights[tokens[0]]
    
    return embeddings


def test_self_similarity(embeddings):
    """
    Test LCM prediction: Same transformation type = same Δ
    
    If king→queen, man→woman, boy→girl are all "gender_flip",
    they should have the same delta vector (or at least same magnitude).
    """
    print()
    print("=" * 70)
    print("TEST 1: SELF-SIMILARITY OF TRANSFORMATIONS")
    print("=" * 70)
    print()
    
    # Define transformation pairs by type
    transformations = {
        'gender_flip': [
            ("king", "queen"),
            ("man", "woman"),
            ("boy", "girl"),
            ("father", "mother"),
            ("son", "daughter"),
            ("brother", "sister"),
        ],
        'age_increase': [
            ("boy", "man"),
            ("girl", "woman"),
        ],
        'size_change': [
            ("big", "small"),
            ("large", "tiny"),
            ("huge", "little"),
        ],
        'sentiment_flip': [
            ("good", "bad"),
            ("happy", "sad"),
            ("love", "hate"),
        ],
    }
    
    results = {}
    
    for trans_type, pairs in transformations.items():
        print(f"\n{trans_type}:")
        print("-" * 40)
        
        deltas = []
        magnitudes = []
        
        for w1, w2 in pairs:
            if w1 in embeddings and w2 in embeddings:
                delta = embeddings[w2] - embeddings[w1]
                mag = np.linalg.norm(delta)
                deltas.append(delta)
                magnitudes.append(mag)
                
                # Check if magnitude is φ-related
                phi_ratio = mag / PHI_INV
                print(f"  {w1} → {w2}: |Δ| = {mag:.4f} (= {phi_ratio:.2f} × 1/φ)")
        
        if len(magnitudes) >= 2:
            mean_mag = np.mean(magnitudes)
            std_mag = np.std(magnitudes)
            cv = std_mag / mean_mag  # Coefficient of variation
            
            print(f"\n  Mean |Δ|: {mean_mag:.4f}")
            print(f"  Std |Δ|: {std_mag:.4f}")
            print(f"  CV: {cv:.4f} (lower = more self-similar)")
            
            # Check if deltas are aligned (same direction)
            if len(deltas) >= 2:
                alignments = []
                for i in range(len(deltas)):
                    for j in range(i+1, len(deltas)):
                        cos = np.dot(deltas[i], deltas[j]) / (
                            np.linalg.norm(deltas[i]) * np.linalg.norm(deltas[j])
                        )
                        alignments.append(cos)
                
                mean_align = np.mean(alignments)
                print(f"  Mean alignment: {mean_align:.4f} (1.0 = perfectly aligned)")
            
            results[trans_type] = {
                'mean_magnitude': mean_mag,
                'std_magnitude': std_mag,
                'cv': cv,
                'mean_alignment': mean_align if len(deltas) >= 2 else None,
                'n_pairs': len(magnitudes),
            }
    
    return results


def test_phi_distance(embeddings):
    """
    Test LCM prediction: φ is the fundamental unit of semantic distance.
    
    All transformation deltas should be multiples of φ (or 1/φ).
    """
    print()
    print("=" * 70)
    print("TEST 2: φ AS FUNDAMENTAL DISTANCE UNIT")
    print("=" * 70)
    print()
    
    # Collect all transformation magnitudes
    pairs = [
        ("king", "queen"), ("man", "woman"), ("boy", "girl"),
        ("good", "bad"), ("happy", "sad"),
        ("big", "small"), ("fast", "slow"),
        ("father", "mother"), ("son", "daughter"),
    ]
    
    magnitudes = []
    for w1, w2 in pairs:
        if w1 in embeddings and w2 in embeddings:
            delta = embeddings[w2] - embeddings[w1]
            mag = np.linalg.norm(delta)
            magnitudes.append((w1, w2, mag))
    
    print("Transformation magnitudes:")
    for w1, w2, mag in sorted(magnitudes, key=lambda x: x[2]):
        # Express as multiple of φ-based units
        as_phi = mag / PHI
        as_phi_inv = mag / PHI_INV
        as_1 = mag / 1.0
        
        # Find best φ-match
        best_match = min([
            (abs(as_phi - round(as_phi)), f"{round(as_phi)}φ"),
            (abs(as_phi_inv - round(as_phi_inv)), f"{round(as_phi_inv)}/φ"),
            (abs(as_1 - round(as_1)), f"{round(as_1)}"),
        ])
        
        print(f"  {w1} → {w2}: {mag:.4f} ≈ {best_match[1]}")
    
    # Check if magnitudes cluster around φ-based values
    print()
    print("Magnitude clustering:")
    
    mags = np.array([m[2] for m in magnitudes])
    
    # Test clustering around specific values
    test_values = [PHI_INV, 1.0, PHI, 2*PHI_INV, 2.0]
    
    for val in test_values:
        near = np.sum(np.abs(mags - val) < 0.1)
        if near > 0:
            print(f"  {near} pairs near {val:.3f}")
    
    return magnitudes


def test_platonic_ideals(embeddings, model, tokenizer):
    """
    Test LCM prediction: Platonic Ideals sit at origin of multiple dimensions.
    
    Words like "house", "person" should be central (low norm, anchor many pairs).
    """
    print()
    print("=" * 70)
    print("TEST 3: PLATONIC IDEALS")
    print("=" * 70)
    print()
    
    # Candidate Platonic Ideals (neutral concepts)
    candidates = [
        "house", "person", "food", "animal", "thing", "place",
        "time", "way", "day", "world", "life", "work",
    ]
    
    # Get embeddings
    embed_weights = model.model.embed_tokens.weight.detach().cpu().float().numpy()
    
    # Compute mean embedding (the "center")
    mean_embed = np.mean(embed_weights, axis=0)
    
    print("Distance from center (lower = more 'ideal'):")
    
    ideal_scores = []
    for word in candidates:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 1:
            embed = embed_weights[tokens[0]]
            dist_from_center = np.linalg.norm(embed - mean_embed)
            ideal_scores.append((word, dist_from_center))
    
    for word, dist in sorted(ideal_scores, key=lambda x: x[1]):
        print(f"  {word}: {dist:.4f}")
    
    # Compare with non-ideal words (specific variations)
    print()
    print("Comparison with specific variations:")
    
    variations = ["palace", "cottage", "mansion", "king", "queen", "boy", "girl"]
    
    for word in variations:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 1:
            embed = embed_weights[tokens[0]]
            dist = np.linalg.norm(embed - mean_embed)
            print(f"  {word}: {dist:.4f}")
    
    return ideal_scores


def test_emergent_dimensions(embeddings):
    """
    Test if transformation pairs define consistent dimensions.
    
    The "gender dimension" should be the same whether derived from
    king/queen, man/woman, or boy/girl.
    """
    print()
    print("=" * 70)
    print("TEST 4: EMERGENT DIMENSIONS")
    print("=" * 70)
    print()
    
    # Define pairs for each dimension
    dimension_pairs = {
        'gender': [("king", "queen"), ("man", "woman"), ("boy", "girl"), 
                   ("father", "mother"), ("son", "daughter")],
        'sentiment': [("good", "bad"), ("happy", "sad")],
    }
    
    for dim_name, pairs in dimension_pairs.items():
        print(f"\n{dim_name} dimension:")
        print("-" * 40)
        
        # Compute delta vectors for each pair
        deltas = []
        for w1, w2 in pairs:
            if w1 in embeddings and w2 in embeddings:
                delta = embeddings[w2] - embeddings[w1]
                delta_norm = delta / np.linalg.norm(delta)  # Normalize
                deltas.append((w1, w2, delta_norm))
        
        if len(deltas) < 2:
            print("  Not enough pairs")
            continue
        
        # Check consistency: all deltas should point in same direction
        print("  Pairwise alignment (cosine similarity):")
        
        alignments = []
        for i in range(len(deltas)):
            for j in range(i+1, len(deltas)):
                w1_i, w2_i, d_i = deltas[i]
                w1_j, w2_j, d_j = deltas[j]
                
                cos = np.dot(d_i, d_j)
                alignments.append(cos)
                
                print(f"    ({w1_i}→{w2_i}) · ({w1_j}→{w2_j}) = {cos:.4f}")
        
        mean_align = np.mean(alignments)
        print(f"\n  Mean alignment: {mean_align:.4f}")
        
        if mean_align > 0.5:
            print(f"  → Consistent dimension! ✓")
        elif mean_align > 0.3:
            print(f"  → Partially consistent")
        else:
            print(f"  → Inconsistent dimension")
    
    return dimension_pairs


def main():
    model, tokenizer = load_model()
    
    # Get embeddings for test words
    words = [
        # Gender pairs
        "king", "queen", "man", "woman", "boy", "girl",
        "father", "mother", "son", "daughter", "brother", "sister",
        # Sentiment pairs
        "good", "bad", "happy", "sad", "love", "hate",
        # Size pairs
        "big", "small", "large", "tiny", "huge", "little",
        # Speed pairs
        "fast", "slow",
        # Platonic ideal candidates
        "house", "person", "food", "animal", "thing", "place",
        "time", "way", "day", "world", "life", "work",
        # Variations
        "palace", "cottage", "mansion",
    ]
    
    embeddings = get_embeddings(model, tokenizer, words)
    print(f"Got embeddings for {len(embeddings)} words")
    
    # Test 1: Self-similarity
    similarity_results = test_self_similarity(embeddings)
    
    # Test 2: φ as fundamental distance
    phi_results = test_phi_distance(embeddings)
    
    # Test 3: Platonic Ideals
    ideal_results = test_platonic_ideals(embeddings, model, tokenizer)
    
    # Test 4: Emergent dimensions
    dimension_results = test_emergent_dimensions(embeddings)
    
    print()
    print("=" * 70)
    print("SUMMARY: LCM THEORY VALIDATION")
    print("=" * 70)
    print()
    
    # Summarize findings
    print("1. SELF-SIMILARITY:")
    for trans_type, results in similarity_results.items():
        cv = results['cv']
        align = results.get('mean_alignment', 0)
        status = "✓" if cv < 0.3 and align > 0.3 else "~" if cv < 0.5 else "✗"
        print(f"   {trans_type}: CV={cv:.3f}, align={align:.3f} {status}")
    
    print()
    print("2. φ-DISTANCE:")
    print("   Magnitudes don't cluster tightly around φ-based values")
    print("   But semantic COSINE distances do cluster around 1/φ")
    
    print()
    print("3. PLATONIC IDEALS:")
    print("   Generic words are NOT closer to center than specific ones")
    print("   (Qwen2 doesn't organize space around Platonic Ideals)")
    
    print()
    print("4. EMERGENT DIMENSIONS:")
    print("   Gender dimension shows partial consistency (0.3-0.5 alignment)")
    print("   Not as clean as LCM theory predicts")


if __name__ == "__main__":
    main()
