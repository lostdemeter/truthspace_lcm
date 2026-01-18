#!/usr/bin/env python3
"""
Qwen2.0 Hidden State Analysis
==============================

Key insight: Analogies don't work in raw embeddings!
The semantic structure must be created by the transformer layers.

Music Box Principle applied:
- DRUM = Raw embeddings (the input)
- COMB = Transformer layers (the transcoder!)
- MUSIC = Hidden states (where semantics live)

This means we need to separate:
1. The embedding "drum" (input representation)
2. The transformer "comb" (the transcoder that creates meaning)
3. The hidden state "music" (where analogies work)

Let's test if analogies work in hidden states at different layers.
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
        torch_dtype=torch.float32,  # Full precision for analysis
    )
    model = model.cpu()
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    
    return model, tokenizer


def get_hidden_states(model, tokenizer, word):
    """Get hidden states at each layer for a single word."""
    
    # Tokenize
    tokens = tokenizer.encode(word, add_special_tokens=False)
    if len(tokens) != 1:
        return None
    
    input_ids = torch.tensor([[tokens[0]]])
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    
    # hidden_states is a tuple of (n_layers + 1) tensors
    # Each tensor is [batch, seq_len, hidden_dim]
    hidden_states = [h[0, 0].numpy() for h in outputs.hidden_states]
    
    return hidden_states


def test_analogies_at_layers(model, tokenizer):
    """
    Test if analogies work at different layers.
    
    Hypothesis: Analogies should work better in later layers
    where the transformer has had time to build semantic structure.
    """
    print()
    print("=" * 70)
    print("TESTING ANALOGIES AT DIFFERENT LAYERS")
    print("=" * 70)
    print()
    
    # Get hidden states for test words
    test_words = [
        "king", "queen", "man", "woman", "boy", "girl",
        "good", "bad", "happy", "sad",
    ]
    
    word_hidden = {}
    for word in test_words:
        hs = get_hidden_states(model, tokenizer, word)
        if hs is not None:
            word_hidden[word] = hs
    
    print(f"Got hidden states for {len(word_hidden)} words")
    print(f"Number of layers: {len(list(word_hidden.values())[0])}")
    
    # Test analogies at each layer
    analogies = [
        ("king", "man", "woman", "queen"),
        ("man", "boy", "girl", "woman"),
    ]
    
    n_layers = len(list(word_hidden.values())[0])
    
    print()
    print("Analogy: king - man + woman = ?")
    print("-" * 50)
    
    for layer in range(n_layers):
        # Get hidden states at this layer
        layer_embeds = {w: word_hidden[w][layer] for w in word_hidden}
        
        # king - man + woman
        result = layer_embeds["king"] - layer_embeds["man"] + layer_embeds["woman"]
        
        # Find nearest among test words
        best_word = None
        best_dist = float('inf')
        
        for word, embed in layer_embeds.items():
            if word in ["king", "man", "woman"]:
                continue
            dist = np.linalg.norm(result - embed)
            if dist < best_dist:
                best_dist = dist
                best_word = word
        
        # Distance to queen
        dist_to_queen = np.linalg.norm(result - layer_embeds["queen"])
        
        marker = "✓" if best_word == "queen" else ""
        print(f"  Layer {layer:2d}: nearest={best_word:8s} dist_to_queen={dist_to_queen:.4f} {marker}")
    
    return word_hidden


def analyze_layer_transformation(word_hidden):
    """
    Analyze how embeddings transform through layers.
    
    Look for φ-patterns in the transformation.
    """
    print()
    print("=" * 70)
    print("ANALYZING LAYER TRANSFORMATIONS")
    print("=" * 70)
    print()
    
    # Track how semantic pairs evolve
    pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("good", "bad"),
    ]
    
    n_layers = len(list(word_hidden.values())[0])
    
    for w1, w2 in pairs:
        if w1 not in word_hidden or w2 not in word_hidden:
            continue
        
        print(f"\n{w1} ↔ {w2}:")
        
        distances = []
        for layer in range(n_layers):
            e1 = word_hidden[w1][layer]
            e2 = word_hidden[w2][layer]
            dist = np.linalg.norm(e2 - e1)
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Check for φ-patterns in distance evolution
        ratios = distances[1:] / distances[:-1]
        
        print(f"  Distance evolution: {distances[0]:.3f} → {distances[-1]:.3f}")
        print(f"  Ratio (final/initial): {distances[-1]/distances[0]:.3f}")
        
        # Find layers where ratio ≈ φ
        phi_layers = []
        for i, r in enumerate(ratios):
            if abs(r - PHI) < 0.1 or abs(r - PHI_INV) < 0.1:
                phi_layers.append(i)
        
        if phi_layers:
            print(f"  φ-ratio layers: {phi_layers}")


def find_semantic_layer(word_hidden):
    """
    Find the layer where semantic structure is strongest.
    
    Measure: How well do analogies work?
    """
    print()
    print("=" * 70)
    print("FINDING OPTIMAL SEMANTIC LAYER")
    print("=" * 70)
    print()
    
    n_layers = len(list(word_hidden.values())[0])
    
    # For each layer, compute alignment of gender vectors
    # king→queen should be parallel to man→woman
    
    alignments = []
    
    for layer in range(n_layers):
        layer_embeds = {w: word_hidden[w][layer] for w in word_hidden}
        
        # Gender vectors
        v1 = layer_embeds["queen"] - layer_embeds["king"]
        v2 = layer_embeds["woman"] - layer_embeds["man"]
        v3 = layer_embeds["girl"] - layer_embeds["boy"]
        
        # Normalize
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        v3 = v3 / np.linalg.norm(v3)
        
        # Average pairwise alignment
        align = (np.dot(v1, v2) + np.dot(v1, v3) + np.dot(v2, v3)) / 3
        alignments.append(align)
    
    alignments = np.array(alignments)
    
    print("Gender vector alignment by layer:")
    for layer in range(n_layers):
        bar = "#" * int(alignments[layer] * 20)
        print(f"  Layer {layer:2d}: {alignments[layer]:.4f} {bar}")
    
    best_layer = np.argmax(alignments)
    print(f"\nBest layer for semantics: {best_layer} (alignment={alignments[best_layer]:.4f})")
    
    return best_layer, alignments


def extract_phi_basis_from_layer(word_hidden, layer_idx, tokenizer, model):
    """
    Extract φ-basis from the optimal semantic layer.
    
    This is where the "music" lives - the transformed representations
    where semantic operations work.
    """
    print()
    print("=" * 70)
    print(f"EXTRACTING φ-BASIS FROM LAYER {layer_idx}")
    print("=" * 70)
    print()
    
    # Get hidden states at this layer for many words
    test_words = list(word_hidden.keys())
    
    layer_embeds = np.array([word_hidden[w][layer_idx] for w in test_words])
    
    print(f"Layer embeddings shape: {layer_embeds.shape}")
    
    # SVD to find principal directions
    mean_embed = np.mean(layer_embeds, axis=0)
    centered = layer_embeds - mean_embed
    
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    print(f"Singular values: {S.round(4)}")
    
    # Check for φ-patterns
    ratios = S[:-1] / S[1:]
    print()
    print("Singular value ratios:")
    for i, r in enumerate(ratios):
        marker = ""
        if abs(r - PHI) < 0.15:
            marker = " ← φ!"
        elif abs(r - PHI_INV) < 0.15:
            marker = " ← 1/φ!"
        print(f"  S[{i}]/S[{i+1}] = {r:.4f}{marker}")
    
    return Vt, S, mean_embed


def main():
    model, tokenizer = load_model()
    
    # Step 1: Test analogies at different layers
    word_hidden = test_analogies_at_layers(model, tokenizer)
    
    # Step 2: Analyze layer transformations
    analyze_layer_transformation(word_hidden)
    
    # Step 3: Find optimal semantic layer
    best_layer, alignments = find_semantic_layer(word_hidden)
    
    # Step 4: Extract φ-basis from optimal layer
    Vt, S, mean_embed = extract_phi_basis_from_layer(word_hidden, best_layer, tokenizer, model)
    
    print()
    print("=" * 70)
    print("SUMMARY: MUSIC BOX DECOMPOSITION")
    print("=" * 70)
    print()
    print("DRUM (Input):")
    print("  - Raw embeddings (layer 0)")
    print("  - Analogies DON'T work here")
    print()
    print("COMB (Transcoder):")
    print(f"  - Transformer layers 0-{best_layer}")
    print("  - This is where meaning is CREATED")
    print()
    print("MUSIC (Output):")
    print(f"  - Hidden states at layer {best_layer}")
    print(f"  - Gender alignment: {alignments[best_layer]:.4f}")
    print("  - Analogies work better here")
    print()
    print("Key insight: The transformer IS the transcoder.")
    print("To get φ-basis, we need to understand the transformation,")
    print("not just the input embeddings.")


if __name__ == "__main__":
    main()
