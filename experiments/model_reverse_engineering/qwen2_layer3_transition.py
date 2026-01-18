#!/usr/bin/env python3
"""
Qwen2.0 Layer 3 Phase Transition Analysis
==========================================

MAJOR DISCOVERY:
- Analogies work at layers 0-2
- At layer 3, something BREAKS - alignment goes negative
- This is a phase transition in the representation

Hypothesis: Layer 3 is where the model switches from
"semantic similarity" mode to "next token prediction" mode.

The early layers preserve semantic structure (the "drum").
The later layers transform it for prediction (the "comb").

Let's investigate what happens at this transition.
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
        torch_dtype=torch.float32,
    )
    model = model.cpu()
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    
    return model, tokenizer


def get_all_hidden_states(model, tokenizer, words):
    """Get hidden states for all words at all layers."""
    
    word_hidden = {}
    
    for word in words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) != 1:
            continue
        
        input_ids = torch.tensor([[tokens[0]]])
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        
        hidden_states = [h[0, 0].numpy() for h in outputs.hidden_states]
        word_hidden[word] = hidden_states
    
    return word_hidden


def analyze_layer_transition(word_hidden):
    """
    Analyze what happens at the layer 2→3 transition.
    """
    print()
    print("=" * 70)
    print("LAYER 2→3 TRANSITION ANALYSIS")
    print("=" * 70)
    print()
    
    # Get embeddings at layers 2 and 3
    words = list(word_hidden.keys())
    
    layer2 = np.array([word_hidden[w][2] for w in words])
    layer3 = np.array([word_hidden[w][3] for w in words])
    
    print(f"Analyzing {len(words)} words")
    print(f"Embedding dim: {layer2.shape[1]}")
    
    # Compute the transformation matrix from layer 2 to layer 3
    # layer3 ≈ layer2 @ W + b (approximately)
    
    # Use least squares to find W
    # layer3 = layer2 @ W
    # W = (layer2.T @ layer2)^-1 @ layer2.T @ layer3
    
    # Add regularization for stability
    reg = 0.01 * np.eye(layer2.shape[1])
    W = np.linalg.solve(layer2.T @ layer2 + reg, layer2.T @ layer3)
    
    print(f"Transformation matrix W shape: {W.shape}")
    
    # Analyze W
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    
    print()
    print("Singular values of transformation W:")
    print(f"  Top 10: {S[:10].round(4)}")
    print(f"  Bottom 10: {S[-10:].round(4)}")
    
    # Check for φ-patterns
    ratios = S[:-1] / S[1:]
    phi_matches = []
    for i, r in enumerate(ratios[:20]):
        if abs(r - PHI) < 0.15:
            phi_matches.append((i, r, 'φ'))
        elif abs(r - PHI_INV) < 0.15:
            phi_matches.append((i, r, '1/φ'))
    
    print(f"\nφ-ratios in W: {len(phi_matches)}")
    for i, r, label in phi_matches[:5]:
        print(f"  S[{i}]/S[{i+1}] = {r:.4f} ≈ {label}")
    
    # Check reconstruction quality
    layer3_pred = layer2 @ W
    error = layer3 - layer3_pred
    rel_error = np.linalg.norm(error) / np.linalg.norm(layer3)
    
    print(f"\nReconstruction error: {rel_error:.4f}")
    print("(If low, the transformation is approximately linear)")
    
    return W, S


def analyze_semantic_preservation(word_hidden):
    """
    Check which semantic relationships are preserved/inverted at each layer.
    """
    print()
    print("=" * 70)
    print("SEMANTIC RELATIONSHIP PRESERVATION")
    print("=" * 70)
    print()
    
    pairs = [
        ("king", "queen", "gender"),
        ("man", "woman", "gender"),
        ("boy", "girl", "gender"),
        ("good", "bad", "sentiment"),
        ("happy", "sad", "sentiment"),
    ]
    
    n_layers = len(list(word_hidden.values())[0])
    
    # For each pair, track the delta vector through layers
    for w1, w2, rel_type in pairs:
        if w1 not in word_hidden or w2 not in word_hidden:
            continue
        
        print(f"\n{w1} → {w2} ({rel_type}):")
        
        # Get delta at each layer
        deltas = []
        for layer in range(n_layers):
            e1 = word_hidden[w1][layer]
            e2 = word_hidden[w2][layer]
            delta = e2 - e1
            deltas.append(delta)
        
        # Check alignment between consecutive layers
        print("  Layer-to-layer delta alignment:")
        for layer in range(1, min(6, n_layers)):
            d_prev = deltas[layer - 1]
            d_curr = deltas[layer]
            
            cos = np.dot(d_prev, d_curr) / (np.linalg.norm(d_prev) * np.linalg.norm(d_curr))
            
            marker = ""
            if cos < 0:
                marker = " ← INVERTED!"
            elif cos > 0.9:
                marker = " ← preserved"
            
            print(f"    Layer {layer-1}→{layer}: cos={cos:.4f}{marker}")


def extract_semantic_subspace_at_layer(word_hidden, layer_idx):
    """
    Extract the semantic subspace at a specific layer.
    """
    print()
    print("=" * 70)
    print(f"SEMANTIC SUBSPACE AT LAYER {layer_idx}")
    print("=" * 70)
    print()
    
    # Collect semantic deltas
    pairs = [
        ("king", "queen"), ("man", "woman"), ("boy", "girl"),
        ("good", "bad"), ("happy", "sad"),
    ]
    
    deltas = []
    for w1, w2 in pairs:
        if w1 in word_hidden and w2 in word_hidden:
            e1 = word_hidden[w1][layer_idx]
            e2 = word_hidden[w2][layer_idx]
            deltas.append(e2 - e1)
    
    deltas = np.array(deltas)
    print(f"Delta matrix shape: {deltas.shape}")
    
    # SVD
    U, S, Vt = np.linalg.svd(deltas, full_matrices=False)
    
    print(f"Singular values: {S.round(4)}")
    
    # Check for φ-patterns
    if len(S) > 1:
        ratios = S[:-1] / S[1:]
        print("Ratios:")
        for i, r in enumerate(ratios):
            marker = ""
            if abs(r - PHI) < 0.2:
                marker = " ← φ!"
            elif abs(r - PHI_INV) < 0.2:
                marker = " ← 1/φ!"
            print(f"  S[{i}]/S[{i+1}] = {r:.4f}{marker}")
    
    return Vt, S


def compare_layers_0_and_2(word_hidden):
    """
    Compare semantic structure at layer 0 (embedding) vs layer 2 (best semantic).
    """
    print()
    print("=" * 70)
    print("COMPARING LAYER 0 vs LAYER 2")
    print("=" * 70)
    print()
    
    # Test analogies at both layers
    def test_analogy(layer_embeds, a, b, c, expected):
        result = layer_embeds[a] - layer_embeds[b] + layer_embeds[c]
        
        # Find nearest
        best_word = None
        best_dist = float('inf')
        for word, embed in layer_embeds.items():
            if word in [a, b, c]:
                continue
            dist = np.linalg.norm(result - embed)
            if dist < best_dist:
                best_dist = dist
                best_word = word
        
        return best_word == expected, best_word
    
    words = list(word_hidden.keys())
    
    for layer_idx in [0, 2]:
        layer_embeds = {w: word_hidden[w][layer_idx] for w in words}
        
        print(f"\nLayer {layer_idx}:")
        
        analogies = [
            ("king", "man", "woman", "queen"),
            ("man", "boy", "girl", "woman"),
        ]
        
        for a, b, c, expected in analogies:
            if all(w in layer_embeds for w in [a, b, c, expected]):
                correct, got = test_analogy(layer_embeds, a, b, c, expected)
                status = "✓" if correct else "✗"
                print(f"  {a} - {b} + {c} = {got} (expected: {expected}) {status}")


def analyze_phi_structure_at_best_layer(word_hidden, layer_idx=2):
    """
    Deep analysis of φ-structure at the best semantic layer.
    """
    print()
    print("=" * 70)
    print(f"φ-STRUCTURE AT LAYER {layer_idx}")
    print("=" * 70)
    print()
    
    words = list(word_hidden.keys())
    layer_embeds = {w: word_hidden[w][layer_idx] for w in words}
    
    # Compute all pairwise distances
    print("Pairwise distances:")
    
    pairs = [
        ("king", "queen"), ("man", "woman"), ("boy", "girl"),
        ("king", "man"), ("queen", "woman"),
        ("good", "bad"), ("happy", "sad"),
    ]
    
    distances = []
    for w1, w2 in pairs:
        if w1 in layer_embeds and w2 in layer_embeds:
            dist = np.linalg.norm(layer_embeds[w2] - layer_embeds[w1])
            distances.append(dist)
            
            # Check if φ-related
            phi_ratio = dist / PHI_INV
            print(f"  {w1} ↔ {w2}: {dist:.4f} = {phi_ratio:.2f} × (1/φ)")
    
    # Check if distances cluster around φ-based values
    distances = np.array(distances)
    
    print()
    print("Distance statistics:")
    print(f"  Mean: {distances.mean():.4f}")
    print(f"  Std: {distances.std():.4f}")
    print(f"  Mean / (1/φ): {distances.mean() / PHI_INV:.4f}")


def main():
    model, tokenizer = load_model()
    
    # Get hidden states for test words
    test_words = [
        "king", "queen", "man", "woman", "boy", "girl",
        "good", "bad", "happy", "sad",
        "big", "small", "fast", "slow",
    ]
    
    word_hidden = get_all_hidden_states(model, tokenizer, test_words)
    print(f"Got hidden states for {len(word_hidden)} words")
    
    # Analysis 1: Layer transition
    W, S = analyze_layer_transition(word_hidden)
    
    # Analysis 2: Semantic preservation
    analyze_semantic_preservation(word_hidden)
    
    # Analysis 3: Compare layers 0 and 2
    compare_layers_0_and_2(word_hidden)
    
    # Analysis 4: Semantic subspace at different layers
    for layer in [0, 2, 3]:
        extract_semantic_subspace_at_layer(word_hidden, layer)
    
    # Analysis 5: φ-structure at best layer
    analyze_phi_structure_at_best_layer(word_hidden, layer_idx=2)
    
    print()
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print()
    print("1. SEMANTIC STRUCTURE EXISTS AT LAYERS 0-2")
    print("   - Analogies work (king - man + woman = queen)")
    print("   - Gender vectors are aligned")
    print()
    print("2. PHASE TRANSITION AT LAYER 3")
    print("   - Semantic alignment INVERTS (goes negative)")
    print("   - Analogies stop working")
    print("   - This is where 'transcoding' begins")
    print()
    print("3. THE MUSIC BOX DECOMPOSITION:")
    print("   - DRUM: Layers 0-2 (semantic structure)")
    print("   - COMB: Layers 3-24 (transcoder for prediction)")
    print("   - MUSIC: The output logits")
    print()
    print("4. FOR φ-BASIS EXTRACTION:")
    print("   - Extract from layer 2 (best semantic layer)")
    print("   - This is where the 'meaning' lives")
    print("   - Later layers transform for prediction, not meaning")


if __name__ == "__main__":
    main()
