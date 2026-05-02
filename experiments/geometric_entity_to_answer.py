#!/usr/bin/env python3
"""
Geometric Entity → Answer: Can We Derive "France → Paris" Without Forward Pass?
================================================================================

This is the KEY experiment for Catch 2 in Doc 181.

The question: Can we derive the answer token from the entity embedding alone,
without running a forward pass through the transformer?

Approaches to test:
1. Linear mapping: answer_emb = W @ entity_emb
2. Rotation: answer_emb = rotate(entity_emb, axis)
3. Offset: answer_emb = entity_emb + offset
4. Analogy: answer_emb = entity_emb + (Paris - France) [learned offset]
5. Platonic Ideal: Navigate via shared ideal

From Doc 180: Rotation axes point toward Platonic Ideals.
From Doc 114: Platonic Ideals sit at origin of multiple dimensions.
From Doc 177: Content tokens require world knowledge (the WALL).

Author: TruthSpace LCM Team
Date: 2026-01-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

PHI = 1.6180339887498949


def get_token_embedding(model, tokenizer, text: str) -> torch.Tensor:
    """Get the embedding for a token/word."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) == 0:
        return None
    # Use first token if multiple
    token_id = tokens[0]
    return model.model.embed_tokens.weight[token_id].detach()


def collect_entity_answer_pairs(model, tokenizer):
    """Collect entity → answer pairs for capitals."""
    
    pairs = [
        # Training pairs
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
        ("Japan", "Tokyo"),
        ("China", "Beijing"),
        ("Russia", "Moscow"),
        ("Brazil", "Brasilia"),
        # Test pairs
        ("Poland", "Warsaw"),
        ("Egypt", "Cairo"),
        ("India", "Delhi"),
        ("Canada", "Ottawa"),
    ]
    
    embeddings = []
    
    for entity, answer in pairs:
        entity_emb = get_token_embedding(model, tokenizer, entity)
        answer_emb = get_token_embedding(model, tokenizer, answer)
        
        if entity_emb is not None and answer_emb is not None:
            embeddings.append({
                'entity': entity,
                'answer': answer,
                'entity_emb': entity_emb,
                'answer_emb': answer_emb,
            })
    
    return embeddings


def test_linear_mapping(embeddings: List[Dict], n_train: int = 8):
    """
    Test 1: Linear mapping W such that answer = W @ entity
    """
    print("\n" + "=" * 70)
    print("Test 1: Linear Mapping (answer = W @ entity)")
    print("=" * 70)
    
    train = embeddings[:n_train]
    test = embeddings[n_train:]
    
    # Stack embeddings
    X = torch.stack([e['entity_emb'] for e in train])  # [n_train, dim]
    Y = torch.stack([e['answer_emb'] for e in train])  # [n_train, dim]
    
    # Solve for W: Y = X @ W.T  =>  W.T = (X.T @ X)^-1 @ X.T @ Y
    # Using least squares
    W, residuals, rank, s = torch.linalg.lstsq(X, Y)
    
    print(f"Training on {n_train} pairs, testing on {len(test)} pairs")
    print(f"Matrix rank: {rank}")
    
    # Test on training data
    print("\n--- Training Data ---")
    train_correct = 0
    for e in train:
        pred_emb = e['entity_emb'] @ W
        # Find closest token
        all_embs = embeddings[0]['entity_emb'].new_zeros(1)  # placeholder
        
        # Compute similarity to actual answer
        sim = F.cosine_similarity(pred_emb.unsqueeze(0), e['answer_emb'].unsqueeze(0))
        print(f"  {e['entity']} → {e['answer']}: similarity = {sim.item():.4f}")
        if sim > 0.5:
            train_correct += 1
    
    # Test on held-out data
    print("\n--- Test Data ---")
    test_correct = 0
    for e in test:
        pred_emb = e['entity_emb'] @ W
        sim = F.cosine_similarity(pred_emb.unsqueeze(0), e['answer_emb'].unsqueeze(0))
        print(f"  {e['entity']} → {e['answer']}: similarity = {sim.item():.4f}")
        if sim > 0.5:
            test_correct += 1
    
    print(f"\nTrain accuracy (sim > 0.5): {train_correct}/{n_train}")
    print(f"Test accuracy (sim > 0.5): {test_correct}/{len(test)}")
    
    return W


def test_offset_approach(embeddings: List[Dict], n_train: int = 8):
    """
    Test 2: Offset approach (answer = entity + offset)
    
    Learn a single offset vector that transforms entity → answer.
    """
    print("\n" + "=" * 70)
    print("Test 2: Offset Approach (answer = entity + offset)")
    print("=" * 70)
    
    train = embeddings[:n_train]
    test = embeddings[n_train:]
    
    # Compute offset for each training pair
    offsets = []
    for e in train:
        offset = e['answer_emb'] - e['entity_emb']
        offsets.append(offset)
    
    # Average offset
    mean_offset = torch.stack(offsets).mean(dim=0)
    
    print(f"Mean offset magnitude: {mean_offset.norm().item():.2f}")
    
    # Test on training data
    print("\n--- Training Data ---")
    train_correct = 0
    for e in train:
        pred_emb = e['entity_emb'] + mean_offset
        sim = F.cosine_similarity(pred_emb.unsqueeze(0), e['answer_emb'].unsqueeze(0))
        print(f"  {e['entity']} → {e['answer']}: similarity = {sim.item():.4f}")
        if sim > 0.5:
            train_correct += 1
    
    # Test on held-out data
    print("\n--- Test Data ---")
    test_correct = 0
    for e in test:
        pred_emb = e['entity_emb'] + mean_offset
        sim = F.cosine_similarity(pred_emb.unsqueeze(0), e['answer_emb'].unsqueeze(0))
        print(f"  {e['entity']} → {e['answer']}: similarity = {sim.item():.4f}")
        if sim > 0.5:
            test_correct += 1
    
    print(f"\nTrain accuracy (sim > 0.5): {train_correct}/{n_train}")
    print(f"Test accuracy (sim > 0.5): {test_correct}/{len(test)}")
    
    return mean_offset


def test_rotation_approach(embeddings: List[Dict], n_train: int = 8):
    """
    Test 3: Rotation approach (answer = rotate(entity, axis))
    
    From Doc 180: Rotation axes point toward Platonic Ideals.
    """
    print("\n" + "=" * 70)
    print("Test 3: Rotation Approach (answer = rotate(entity, axis))")
    print("=" * 70)
    
    train = embeddings[:n_train]
    test = embeddings[n_train:]
    
    # For each training pair, compute the rotation that maps entity → answer
    # Using Rodrigues' rotation formula
    
    # First, let's see if there's a consistent rotation axis
    axes = []
    angles = []
    
    for e in train:
        entity_norm = e['entity_emb'] / e['entity_emb'].norm()
        answer_norm = e['answer_emb'] / e['answer_emb'].norm()
        
        # Rotation axis = entity × answer (cross product in high-D is tricky)
        # Instead, use the plane spanned by entity and answer
        
        # Angle between them
        cos_angle = (entity_norm * answer_norm).sum()
        angle = torch.acos(cos_angle.clamp(-1, 1))
        angles.append(angle.item())
        
        # Direction of rotation (in the plane)
        # answer = cos(θ) * entity + sin(θ) * perpendicular
        perp = answer_norm - cos_angle * entity_norm
        if perp.norm() > 1e-6:
            perp = perp / perp.norm()
        axes.append(perp)
    
    print(f"Angles (radians): {[f'{a:.3f}' for a in angles]}")
    print(f"Mean angle: {np.mean(angles):.3f} rad = {np.degrees(np.mean(angles)):.1f}°")
    
    # Check if axes are similar
    if len(axes) >= 2:
        axis_sims = []
        for i in range(len(axes)):
            for j in range(i+1, len(axes)):
                sim = (axes[i] * axes[j]).sum().abs().item()
                axis_sims.append(sim)
        print(f"Axis similarities: mean={np.mean(axis_sims):.3f}, std={np.std(axis_sims):.3f}")
    
    # Use mean angle and try to apply rotation
    mean_angle = np.mean(angles)
    
    print("\n--- Training Data ---")
    train_correct = 0
    for i, e in enumerate(train):
        entity_norm = e['entity_emb'] / e['entity_emb'].norm()
        
        # Rotate by mean angle in the direction of this pair's axis
        pred_emb = np.cos(mean_angle) * e['entity_emb'] + np.sin(mean_angle) * axes[i] * e['entity_emb'].norm()
        
        sim = F.cosine_similarity(pred_emb.unsqueeze(0), e['answer_emb'].unsqueeze(0))
        print(f"  {e['entity']} → {e['answer']}: similarity = {sim.item():.4f}")
        if sim > 0.5:
            train_correct += 1
    
    print(f"\nTrain accuracy (sim > 0.5): {train_correct}/{n_train}")
    
    return mean_angle, axes


def test_analogy_approach(embeddings: List[Dict], n_train: int = 8, model=None, tokenizer=None):
    """
    Test 4: Analogy approach (answer = entity + (Paris - France))
    
    Classic word2vec style analogy.
    """
    print("\n" + "=" * 70)
    print("Test 4: Analogy Approach (answer = entity + (Paris - France))")
    print("=" * 70)
    
    train = embeddings[:n_train]
    test = embeddings[n_train:]
    
    # Use first pair as the reference
    ref = train[0]
    capital_offset = ref['answer_emb'] - ref['entity_emb']
    
    print(f"Reference: {ref['entity']} → {ref['answer']}")
    print(f"Offset magnitude: {capital_offset.norm().item():.2f}")
    
    # Test on other training data
    print("\n--- Training Data (excluding reference) ---")
    train_correct = 0
    for e in train[1:]:
        pred_emb = e['entity_emb'] + capital_offset
        sim = F.cosine_similarity(pred_emb.unsqueeze(0), e['answer_emb'].unsqueeze(0))
        print(f"  {e['entity']} → {e['answer']}: similarity = {sim.item():.4f}")
        if sim > 0.5:
            train_correct += 1
    
    # Test on held-out data
    print("\n--- Test Data ---")
    test_correct = 0
    for e in test:
        pred_emb = e['entity_emb'] + capital_offset
        sim = F.cosine_similarity(pred_emb.unsqueeze(0), e['answer_emb'].unsqueeze(0))
        print(f"  {e['entity']} → {e['answer']}: similarity = {sim.item():.4f}")
        if sim > 0.5:
            test_correct += 1
    
    print(f"\nTrain accuracy (sim > 0.5): {train_correct}/{n_train-1}")
    print(f"Test accuracy (sim > 0.5): {test_correct}/{len(test)}")
    
    # Now test if we can actually decode to the right token
    if model is not None:
        print("\n--- Decoding Test ---")
        lm_head = model.lm_head.weight.data
        
        for e in test:
            pred_emb = e['entity_emb'] + capital_offset
            
            # Decode
            logits = pred_emb @ lm_head.T
            top_tokens = logits.topk(5).indices
            top_words = [tokenizer.decode([t]) for t in top_tokens]
            
            print(f"  {e['entity']} → predicted: {top_words}, actual: {e['answer']}")
    
    return capital_offset


def test_platonic_ideal_approach(embeddings: List[Dict], n_train: int = 8):
    """
    Test 5: Platonic Ideal approach
    
    From Doc 114: Platonic Ideals sit at the origin of multiple dimensions.
    
    Hypothesis: There's a "capital" ideal, and both entity and answer
    are related to it.
    """
    print("\n" + "=" * 70)
    print("Test 5: Platonic Ideal Approach")
    print("=" * 70)
    
    train = embeddings[:n_train]
    test = embeddings[n_train:]
    
    # Compute the "capital ideal" as the centroid of all entity-answer midpoints
    midpoints = []
    for e in train:
        midpoint = (e['entity_emb'] + e['answer_emb']) / 2
        midpoints.append(midpoint)
    
    capital_ideal = torch.stack(midpoints).mean(dim=0)
    
    print(f"Capital ideal magnitude: {capital_ideal.norm().item():.2f}")
    
    # For each entity, the answer should be on the opposite side of the ideal
    # answer = 2 * ideal - entity (reflection through ideal)
    
    print("\n--- Training Data ---")
    train_correct = 0
    for e in train:
        pred_emb = 2 * capital_ideal - e['entity_emb']
        sim = F.cosine_similarity(pred_emb.unsqueeze(0), e['answer_emb'].unsqueeze(0))
        print(f"  {e['entity']} → {e['answer']}: similarity = {sim.item():.4f}")
        if sim > 0.5:
            train_correct += 1
    
    print("\n--- Test Data ---")
    test_correct = 0
    for e in test:
        pred_emb = 2 * capital_ideal - e['entity_emb']
        sim = F.cosine_similarity(pred_emb.unsqueeze(0), e['answer_emb'].unsqueeze(0))
        print(f"  {e['entity']} → {e['answer']}: similarity = {sim.item():.4f}")
        if sim > 0.5:
            test_correct += 1
    
    print(f"\nTrain accuracy (sim > 0.5): {train_correct}/{n_train}")
    print(f"Test accuracy (sim > 0.5): {test_correct}/{len(test)}")
    
    return capital_ideal


def test_hidden_state_approach(model, tokenizer, embeddings: List[Dict], n_train: int = 8):
    """
    Test 6: Hidden state approach
    
    Instead of using raw embeddings, use the hidden state after processing
    the prompt "The capital of [entity] is".
    
    This is closer to what the model actually does.
    """
    print("\n" + "=" * 70)
    print("Test 6: Hidden State Approach (after prompt processing)")
    print("=" * 70)
    
    train = embeddings[:n_train]
    test = embeddings[n_train:]
    
    # Collect hidden states
    hidden_states = []
    
    for e in embeddings:
        prompt = f"The capital of {e['entity']} is"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]  # Last layer, last token
        
        hidden_states.append({
            'entity': e['entity'],
            'answer': e['answer'],
            'hidden': h,
            'answer_emb': e['answer_emb'],
        })
    
    train_h = hidden_states[:n_train]
    test_h = hidden_states[n_train:]
    
    # Learn mapping from hidden state to answer embedding
    X = torch.stack([h['hidden'] for h in train_h])
    Y = torch.stack([h['answer_emb'] for h in train_h])
    
    # Linear mapping
    W, residuals, rank, s = torch.linalg.lstsq(X, Y)
    
    print(f"Hidden state dim: {X.shape[1]}")
    print(f"Answer emb dim: {Y.shape[1]}")
    
    # Test
    lm_head = model.lm_head.weight.data
    
    print("\n--- Training Data ---")
    train_correct = 0
    for h in train_h:
        pred_emb = h['hidden'] @ W
        
        # Decode
        logits = pred_emb @ lm_head.T
        pred_token = logits.argmax().item()
        pred_word = tokenizer.decode([pred_token])
        
        correct = pred_word.strip().lower() == h['answer'].lower()
        if correct:
            train_correct += 1
        
        print(f"  {h['entity']} → predicted: '{pred_word}', actual: '{h['answer']}' {'✓' if correct else '✗'}")
    
    print("\n--- Test Data ---")
    test_correct = 0
    for h in test_h:
        pred_emb = h['hidden'] @ W
        
        # Decode
        logits = pred_emb @ lm_head.T
        pred_token = logits.argmax().item()
        pred_word = tokenizer.decode([pred_token])
        
        correct = pred_word.strip().lower() == h['answer'].lower()
        if correct:
            test_correct += 1
        
        print(f"  {h['entity']} → predicted: '{pred_word}', actual: '{h['answer']}' {'✓' if correct else '✗'}")
    
    print(f"\nTrain accuracy: {train_correct}/{n_train} = {train_correct/n_train*100:.1f}%")
    print(f"Test accuracy: {test_correct}/{len(test_h)} = {test_correct/len(test_h)*100:.1f}%")
    
    return W


def synthesize_results():
    """Synthesize findings."""
    print("\n" + "=" * 70)
    print("SYNTHESIS: Geometric Entity → Answer")
    print("=" * 70)
    print("""
The question: Can we derive "France → Paris" without a forward pass?

APPROACHES TESTED:
==================

1. Linear Mapping (answer = W @ entity)
   - Learns a transformation matrix
   - Works on training, may not generalize

2. Offset (answer = entity + offset)
   - Single offset vector for all pairs
   - Simple but may not capture variation

3. Rotation (answer = rotate(entity, axis))
   - From Doc 180: rotation axes point to Platonic Ideals
   - Requires consistent rotation axis

4. Analogy (answer = entity + (Paris - France))
   - Classic word2vec approach
   - Uses one pair as reference

5. Platonic Ideal (answer = 2*ideal - entity)
   - Reflection through the "capital" ideal
   - From Doc 114

6. Hidden State (after prompt processing)
   - Uses hidden state, not raw embedding
   - Closer to what model actually does

KEY INSIGHT:
============

The raw embedding approach (Tests 1-5) operates on TOKEN embeddings.
But the model doesn't just use token embeddings - it processes the
entire prompt through 28 layers of attention and MLP.

The hidden state approach (Test 6) is more realistic because it
captures what the model actually computes.

IMPLICATIONS FOR CATCH 2:
=========================

If hidden state approach works:
  - We still need ONE forward pass to get the hidden state
  - But we can then decode ALL tokens geometrically
  - This is what we already have!

If raw embedding approach works:
  - We can skip the forward pass entirely
  - Entity embedding → answer embedding directly
  - This would be TRUE geometric speedup

The gap between these approaches IS the transformer's computation.
""")


def main():
    print("=" * 70)
    print("Geometric Entity → Answer: Testing Catch 2 Solutions")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Collect entity-answer pairs
    embeddings = collect_entity_answer_pairs(model, tokenizer)
    print(f"\nCollected {len(embeddings)} entity-answer pairs")
    for e in embeddings:
        print(f"  {e['entity']} → {e['answer']}")
    
    # Run tests
    test_linear_mapping(embeddings)
    test_offset_approach(embeddings)
    test_rotation_approach(embeddings)
    test_analogy_approach(embeddings, model=model, tokenizer=tokenizer)
    test_platonic_ideal_approach(embeddings)
    test_hidden_state_approach(model, tokenizer, embeddings)
    
    # Synthesis
    synthesize_results()


if __name__ == "__main__":
    main()
