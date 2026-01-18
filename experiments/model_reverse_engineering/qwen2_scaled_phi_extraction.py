#!/usr/bin/env python3
"""
Qwen2.0 Scaled φ-Basis Extraction
==================================

Scale the φ-extraction to a larger vocabulary sample to verify
that the patterns hold at scale.

Test with 500+ words across different categories:
- Common words (high frequency)
- Semantic pairs (gender, sentiment, size)
- Verbs (tense variations)
- Abstract concepts
- Proper nouns (if single-token)
"""

import torch
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


def load_model():
    """Load Qwen2-0.5B model."""
    print("Loading Qwen2-0.5B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
        torch_dtype=torch.float32,
    )
    model = model.cpu()
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    
    return model, tokenizer


def get_test_vocabulary():
    """Get a diverse test vocabulary."""
    
    words = []
    
    # Gender pairs
    words.extend([
        "king", "queen", "man", "woman", "boy", "girl",
        "father", "mother", "son", "daughter", "brother", "sister",
        "uncle", "aunt", "nephew", "niece", "husband", "wife",
        "prince", "princess", "actor", "actress", "waiter", "waitress",
        "he", "she", "him", "her", "his", "hers",
    ])
    
    # Sentiment
    words.extend([
        "good", "bad", "happy", "sad", "love", "hate",
        "beautiful", "ugly", "nice", "mean", "kind", "cruel",
        "brave", "coward", "smart", "dumb", "rich", "poor",
        "strong", "weak", "fast", "slow", "hot", "cold",
        "light", "dark", "clean", "dirty", "safe", "dangerous",
    ])
    
    # Size
    words.extend([
        "big", "small", "large", "tiny", "huge", "little",
        "giant", "dwarf", "tall", "short", "wide", "narrow",
        "thick", "thin", "deep", "shallow", "long", "brief",
    ])
    
    # Common verbs
    words.extend([
        "go", "went", "gone", "going",
        "is", "was", "be", "been", "being",
        "have", "had", "has", "having",
        "do", "did", "done", "doing",
        "say", "said", "saying",
        "make", "made", "making",
        "know", "knew", "known", "knowing",
        "think", "thought", "thinking",
        "take", "took", "taken", "taking",
        "see", "saw", "seen", "seeing",
        "come", "came", "coming",
        "want", "wanted", "wanting",
        "use", "used", "using",
        "find", "found", "finding",
        "give", "gave", "given", "giving",
    ])
    
    # Common nouns
    words.extend([
        "time", "year", "people", "way", "day", "thing",
        "world", "life", "hand", "part", "place", "case",
        "week", "company", "system", "program", "question",
        "work", "government", "number", "night", "point",
        "home", "water", "room", "mother", "area", "money",
        "story", "fact", "month", "lot", "right", "study",
        "book", "eye", "job", "word", "business", "issue",
        "side", "kind", "head", "house", "service", "friend",
        "power", "hour", "game", "line", "end", "member",
    ])
    
    # Abstract concepts
    words.extend([
        "truth", "love", "peace", "war", "freedom", "justice",
        "hope", "fear", "anger", "joy", "pain", "pleasure",
        "knowledge", "wisdom", "power", "beauty", "evil",
        "faith", "doubt", "reason", "emotion", "thought",
    ])
    
    # Numbers
    words.extend([
        "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
        "first", "second", "third", "last",
    ])
    
    # Colors
    words.extend([
        "red", "blue", "green", "yellow", "black", "white",
        "orange", "purple", "pink", "brown", "gray",
    ])
    
    # Time words
    words.extend([
        "now", "then", "today", "tomorrow", "yesterday",
        "always", "never", "sometimes", "often", "rarely",
        "before", "after", "during", "while", "until",
    ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique_words.append(w)
    
    return unique_words


def get_layer2_embeddings_batch(model, tokenizer, words, batch_size=50):
    """Get Layer 2 hidden states for many words efficiently."""
    
    embeddings = {}
    valid_words = []
    
    # First, filter to single-token words
    for word in words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 1:
            valid_words.append((word, tokens[0]))
    
    print(f"Found {len(valid_words)} single-token words out of {len(words)}")
    
    # Process in batches
    for i in range(0, len(valid_words), batch_size):
        batch = valid_words[i:i+batch_size]
        
        # Create input tensor
        input_ids = torch.tensor([[t[1]] for t in batch])
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        
        # Extract Layer 2 hidden states
        layer2 = outputs.hidden_states[2]  # [batch, 1, hidden_dim]
        
        for j, (word, _) in enumerate(batch):
            embeddings[word] = layer2[j, 0].numpy()
        
        if (i + batch_size) % 100 == 0:
            print(f"  Processed {min(i + batch_size, len(valid_words))}/{len(valid_words)} words")
    
    return embeddings


def extract_phi_basis_scaled(embeddings):
    """Extract φ-basis from scaled embeddings."""
    print()
    print("=" * 70)
    print(f"EXTRACTING φ-BASIS FROM {len(embeddings)} WORDS")
    print("=" * 70)
    print()
    
    words = list(embeddings.keys())
    embed_matrix = np.array([embeddings[w] for w in words])
    
    print(f"Embedding matrix shape: {embed_matrix.shape}")
    
    # Center embeddings
    mean_embed = np.mean(embed_matrix, axis=0)
    centered = embed_matrix - mean_embed
    
    # SVD
    print("Computing SVD...")
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    print(f"Singular values shape: {S.shape}")
    print(f"Top 20 singular values: {S[:20].round(2)}")
    
    # Check for φ-patterns
    ratios = S[:-1] / S[1:]
    
    print()
    print("Singular value ratios (looking for φ ≈ 1.618):")
    phi_matches = []
    for i in range(min(30, len(ratios))):
        r = ratios[i]
        if abs(r - PHI) < 0.15:
            phi_matches.append((i, r, 'φ'))
            print(f"  S[{i}]/S[{i+1}] = {r:.4f} ← φ!")
        elif abs(r - PHI_INV) < 0.15:
            phi_matches.append((i, r, '1/φ'))
            print(f"  S[{i}]/S[{i+1}] = {r:.4f} ← 1/φ!")
    
    print(f"\nTotal φ-matches in top 30: {len(phi_matches)}")
    
    # Cumulative variance
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    
    print()
    print("Cumulative variance:")
    for thresh in [0.5, 0.8, 0.9, 0.95, 0.99]:
        n = np.searchsorted(cumvar, thresh) + 1
        print(f"  {thresh*100:.0f}%: {n} dimensions")
    
    return U, S, Vt, mean_embed, words


def test_reconstruction_scaled(embeddings, S, Vt, mean_embed, words):
    """Test reconstruction at different dimension counts."""
    print()
    print("=" * 70)
    print("TESTING RECONSTRUCTION AT DIFFERENT DIMENSIONS")
    print("=" * 70)
    print()
    
    embed_matrix = np.array([embeddings[w] for w in words])
    centered = embed_matrix - mean_embed
    
    # φ-weighting
    def get_phi_weights(n_dims):
        k = n_dims / 5
        return np.array([PHI ** (-i / k) for i in range(n_dims)])
    
    results = []
    
    for n_dims in [10, 20, 50, 100, 200, min(len(S), 500)]:
        if n_dims > len(S):
            continue
        
        # Truncated basis
        Vt_trunc = Vt[:n_dims]
        phi_weights = get_phi_weights(n_dims)
        
        # Project
        coords = centered @ Vt_trunc.T
        coords_weighted = coords * phi_weights
        
        # Reconstruct
        coords_unweighted = coords_weighted / phi_weights
        reconstructed = coords_unweighted @ Vt_trunc + mean_embed
        
        # Error
        error = embed_matrix - reconstructed
        rel_error = np.linalg.norm(error) / np.linalg.norm(embed_matrix)
        var_explained = 1 - rel_error**2
        
        results.append({
            'n_dims': n_dims,
            'rel_error': rel_error,
            'var_explained': var_explained,
        })
        
        print(f"  {n_dims:4d} dims: error={rel_error:.6f}, variance={var_explained:.4%}")
    
    return results


def test_analogies_scaled(embeddings, Vt, mean_embed, n_dims=50):
    """Test analogies at scale."""
    print()
    print("=" * 70)
    print(f"TESTING ANALOGIES ({n_dims} dimensions)")
    print("=" * 70)
    print()
    
    words = list(embeddings.keys())
    embed_matrix = np.array([embeddings[w] for w in words])
    centered = embed_matrix - mean_embed
    
    # Project to subspace
    Vt_trunc = Vt[:n_dims]
    k = n_dims / 5
    phi_weights = np.array([PHI ** (-i / k) for i in range(n_dims)])
    
    coords = centered @ Vt_trunc.T
    coords_weighted = coords * phi_weights
    
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    def get_coords(word):
        if word in word_to_idx:
            return coords_weighted[word_to_idx[word]]
        return None
    
    def find_nearest(target_coords, exclude=[]):
        distances = np.linalg.norm(coords_weighted - target_coords, axis=1)
        for w in exclude:
            if w in word_to_idx:
                distances[word_to_idx[w]] = np.inf
        nearest_idx = np.argmin(distances)
        return words[nearest_idx]
    
    # Test analogies
    analogies = [
        ("king", "man", "woman", "queen"),
        ("man", "boy", "girl", "woman"),
        ("father", "mother", "son", "daughter"),
        ("good", "bad", "happy", "sad"),
        ("big", "small", "tall", "short"),
        ("go", "went", "come", "came"),
        ("is", "was", "do", "did"),
    ]
    
    correct = 0
    total = 0
    
    for a, b, c, expected in analogies:
        ca, cb, cc = get_coords(a), get_coords(b), get_coords(c)
        
        if any(x is None for x in [ca, cb, cc]) or expected not in word_to_idx:
            continue
        
        total += 1
        result = ca - cb + cc
        nearest = find_nearest(result, exclude=[a, b, c])
        
        if nearest == expected:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"  {a} - {b} + {c} = {nearest} (expected: {expected}) {status}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1%}")
    
    return accuracy


def analyze_coordinate_distribution(embeddings, Vt, mean_embed, n_dims=50):
    """Analyze the distribution of φ-coordinates."""
    print()
    print("=" * 70)
    print("φ-COORDINATE DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print()
    
    words = list(embeddings.keys())
    embed_matrix = np.array([embeddings[w] for w in words])
    centered = embed_matrix - mean_embed
    
    # Project
    Vt_trunc = Vt[:n_dims]
    k = n_dims / 5
    phi_weights = np.array([PHI ** (-i / k) for i in range(n_dims)])
    
    coords = centered @ Vt_trunc.T
    coords_weighted = coords * phi_weights
    
    all_coords = coords_weighted.flatten()
    
    print(f"Total coordinates: {len(all_coords)}")
    print(f"Mean: {all_coords.mean():.4f}")
    print(f"Std: {all_coords.std():.4f}")
    print(f"Range: [{all_coords.min():.4f}, {all_coords.max():.4f}]")
    
    # Check clustering around φ-based values
    print()
    print("Clustering around φ-based values:")
    
    test_values = [
        (0, "0"),
        (PHI_INV, "1/φ"),
        (-PHI_INV, "-1/φ"),
        (1, "1"),
        (-1, "-1"),
        (PHI, "φ"),
        (-PHI, "-φ"),
        (PHI**2, "φ²"),
        (-PHI**2, "-φ²"),
    ]
    
    for val, name in test_values:
        # Count within 0.3 of value
        near = np.sum(np.abs(all_coords - val) < 0.3)
        pct = near / len(all_coords) * 100
        if pct > 1:
            print(f"  Near {name:5s} ({val:+.3f}): {pct:.1f}%")
    
    return coords_weighted


def main():
    model, tokenizer = load_model()
    
    # Get test vocabulary
    test_words = get_test_vocabulary()
    print(f"Test vocabulary: {len(test_words)} words")
    
    # Get Layer 2 embeddings
    embeddings = get_layer2_embeddings_batch(model, tokenizer, test_words)
    
    # Extract φ-basis
    U, S, Vt, mean_embed, words = extract_phi_basis_scaled(embeddings)
    
    # Test reconstruction
    results = test_reconstruction_scaled(embeddings, S, Vt, mean_embed, words)
    
    # Test analogies at different dimensions
    print()
    print("Testing analogies at different dimension counts:")
    for n_dims in [10, 20, 50, 100]:
        if n_dims <= len(S):
            acc = test_analogies_scaled(embeddings, Vt, mean_embed, n_dims)
    
    # Analyze coordinate distribution
    coords = analyze_coordinate_distribution(embeddings, Vt, mean_embed, n_dims=50)
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Vocabulary size: {len(embeddings)} words")
    print(f"Embedding dimension: {Vt.shape[1]}")
    print(f"φ-basis dimensions: {len(S)}")
    print()
    print("Key findings at scale:")
    print("1. φ-patterns in singular value ratios")
    print("2. Reconstruction quality vs dimensions")
    print("3. Analogy accuracy vs dimensions")
    print("4. Coordinate clustering around φ-values")


if __name__ == "__main__":
    main()
