#!/usr/bin/env python3
"""
Upcast Generalization V2: Direct Linear Mapping
================================================

The previous approach failed because we used the "similar dimensions"
projection which was specific to the training pairs.

New approach: Learn a DIRECT linear mapping from entity to answer.
  answer_embed = entity_embed @ W

This is simpler and might generalize better.

Author: TruthSpace LCM Team
Date: 2026-01-30
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


def test_direct_linear_mapping(model, tokenizer):
    """
    Test direct linear mapping: answer = entity @ W
    """
    print("\n" + "=" * 70)
    print("Direct Linear Mapping: answer = entity @ W")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    
    # Training pairs
    train_pairs = [
        ("France", " Paris"),
        ("Germany", " Berlin"),
        ("Italy", " Rome"),
        ("Spain", " Madrid"),
        ("Japan", " Tokyo"),
        ("China", " Beijing"),
        ("Poland", " Warsaw"),
        ("Russia", " Moscow"),
        ("Canada", " Ottawa"),
        ("Australia", " Canberra"),
    ]
    
    # Test pairs
    test_pairs = [
        ("Sweden", " Stockholm"),
        ("Norway", " Oslo"),
        ("Finland", " Helsinki"),
        ("Denmark", " Copenhagen"),
        ("Netherlands", " Amsterdam"),
        ("Belgium", " Brussels"),
        ("Austria", " Vienna"),
        ("Switzerland", " Bern"),
        ("Portugal", " Lisbon"),
        ("Greece", " Athens"),
    ]
    
    # Get embeddings
    train_entities = []
    train_answers = []
    
    for entity, answer in train_pairs:
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        
        train_entities.append(embed[entity_ids[0]])
        train_answers.append(embed[answer_ids[0]])
    
    E_train = torch.stack(train_entities)
    A_train = torch.stack(train_answers)
    
    print(f"Training on {len(train_pairs)} pairs")
    
    # Learn W: A = E @ W
    lambda_reg = 0.1
    EtE = E_train.T @ E_train + lambda_reg * torch.eye(E_train.shape[1])
    EtA = E_train.T @ A_train
    W = torch.linalg.solve(EtE, EtA)
    
    print(f"W shape: {W.shape}")
    
    # Test on training data
    print("\n--- Training Data ---")
    
    train_correct = 0
    for i, (entity, answer) in enumerate(train_pairs):
        a_pred = E_train[i] @ W
        
        sims = F.cosine_similarity(a_pred.unsqueeze(0), embed)
        pred_idx = sims.argmax()
        pred_text = tokenizer.decode([pred_idx])
        
        marker = "✓" if pred_text.strip() == answer.strip() else "✗"
        if pred_text.strip() == answer.strip():
            train_correct += 1
        
        print(f"  {entity} → pred: {pred_text!r}, expected: {answer!r} {marker}")
    
    print(f"  Training accuracy: {train_correct}/{len(train_pairs)}")
    
    # Test on unseen data
    print("\n--- Unseen Data ---")
    
    test_correct = 0
    for entity, expected in test_pairs:
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        entity_embed = embed[entity_ids[0]]
        
        a_pred = entity_embed @ W
        
        sims = F.cosine_similarity(a_pred.unsqueeze(0), embed)
        pred_idx = sims.argmax()
        pred_text = tokenizer.decode([pred_idx])
        
        # Get true answer from transformer
        prompt = f"The capital of {entity} is"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0))
            true_idx = outputs.logits[0, -1, :].argmax()
        true_text = tokenizer.decode([true_idx])
        
        marker = "✓" if pred_text.strip() == true_text.strip() else "✗"
        if pred_text.strip() == true_text.strip():
            test_correct += 1
        
        print(f"  {entity} → pred: {pred_text!r}, true: {true_text!r} {marker}")
    
    print(f"\n  Generalization accuracy: {test_correct}/{len(test_pairs)} = {test_correct/len(test_pairs)*100:.1f}%")
    
    return test_correct / len(test_pairs)


def test_offset_approach(model, tokenizer):
    """
    Test offset approach: answer = entity + offset
    
    Where offset is learned from training pairs.
    """
    print("\n" + "=" * 70)
    print("Offset Approach: answer = entity + offset")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    
    # Training pairs
    train_pairs = [
        ("France", " Paris"),
        ("Germany", " Berlin"),
        ("Italy", " Rome"),
        ("Spain", " Madrid"),
        ("Japan", " Tokyo"),
        ("China", " Beijing"),
        ("Poland", " Warsaw"),
        ("Russia", " Moscow"),
    ]
    
    # Test pairs
    test_pairs = [
        ("Sweden", " Stockholm"),
        ("Norway", " Oslo"),
        ("Finland", " Helsinki"),
        ("Denmark", " Copenhagen"),
        ("Austria", " Vienna"),
    ]
    
    # Get embeddings
    train_entities = []
    train_answers = []
    
    for entity, answer in train_pairs:
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        
        train_entities.append(embed[entity_ids[0]])
        train_answers.append(embed[answer_ids[0]])
    
    E_train = torch.stack(train_entities)
    A_train = torch.stack(train_answers)
    
    # Compute offsets
    offsets = A_train - E_train
    
    # Mean offset
    mean_offset = offsets.mean(dim=0)
    
    print(f"Mean offset norm: {mean_offset.norm():.4f}")
    print(f"Individual offset norms: {[o.norm().item() for o in offsets]}")
    
    # Test with mean offset
    print("\n--- Test with Mean Offset ---")
    
    test_correct = 0
    for entity, expected in test_pairs:
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        entity_embed = embed[entity_ids[0]]
        
        a_pred = entity_embed + mean_offset
        
        sims = F.cosine_similarity(a_pred.unsqueeze(0), embed)
        pred_idx = sims.argmax()
        pred_text = tokenizer.decode([pred_idx])
        
        # Get true answer
        prompt = f"The capital of {entity} is"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0))
            true_idx = outputs.logits[0, -1, :].argmax()
        true_text = tokenizer.decode([true_idx])
        
        marker = "✓" if pred_text.strip() == true_text.strip() else "✗"
        if pred_text.strip() == true_text.strip():
            test_correct += 1
        
        print(f"  {entity} → pred: {pred_text!r}, true: {true_text!r} {marker}")
    
    print(f"\n  Generalization accuracy: {test_correct}/{len(test_pairs)} = {test_correct/len(test_pairs)*100:.1f}%")


def test_analogy_approach(model, tokenizer):
    """
    Test analogy approach: France:Paris :: Germany:?
    
    answer = entity - France + Paris
    """
    print("\n" + "=" * 70)
    print("Analogy Approach: answer = entity - France + Paris")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    
    # Reference pair
    ref_entity = "France"
    ref_answer = " Paris"
    
    ref_entity_ids = tokenizer.encode(ref_entity, add_special_tokens=False)
    ref_answer_ids = tokenizer.encode(ref_answer, add_special_tokens=False)
    
    ref_entity_embed = embed[ref_entity_ids[0]]
    ref_answer_embed = embed[ref_answer_ids[0]]
    
    # Test pairs
    test_pairs = [
        ("Germany", " Berlin"),
        ("Italy", " Rome"),
        ("Spain", " Madrid"),
        ("Japan", " Tokyo"),
        ("Sweden", " Stockholm"),
        ("Norway", " Oslo"),
        ("Austria", " Vienna"),
    ]
    
    print(f"Reference: {ref_entity} → {ref_answer}")
    print("\n--- Analogy Test ---")
    
    test_correct = 0
    for entity, expected in test_pairs:
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        entity_embed = embed[entity_ids[0]]
        
        # Analogy: entity - France + Paris
        a_pred = entity_embed - ref_entity_embed + ref_answer_embed
        
        sims = F.cosine_similarity(a_pred.unsqueeze(0), embed)
        pred_idx = sims.argmax()
        pred_text = tokenizer.decode([pred_idx])
        
        # Get true answer
        prompt = f"The capital of {entity} is"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0))
            true_idx = outputs.logits[0, -1, :].argmax()
        true_text = tokenizer.decode([true_idx])
        
        marker = "✓" if pred_text.strip() == true_text.strip() else "✗"
        if pred_text.strip() == true_text.strip():
            test_correct += 1
        
        print(f"  {entity} → pred: {pred_text!r}, true: {true_text!r} {marker}")
    
    print(f"\n  Generalization accuracy: {test_correct}/{len(test_pairs)} = {test_correct/len(test_pairs)*100:.1f}%")


def test_hidden_state_approach(model, tokenizer):
    """
    The KEY insight: We should use HIDDEN STATES, not embeddings!
    
    The transformer transforms embeddings into hidden states.
    The hidden state is what predicts the next token.
    
    Maybe the relationship is in hidden state space, not embedding space.
    """
    print("\n" + "=" * 70)
    print("Hidden State Approach")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    lm_head = model.lm_head.weight.data
    
    # Training pairs with prompts
    train_data = [
        ("The capital of France is", " Paris"),
        ("The capital of Germany is", " Berlin"),
        ("The capital of Italy is", " Rome"),
        ("The capital of Spain is", " Madrid"),
        ("The capital of Japan is", " Tokyo"),
        ("The capital of China is", " Beijing"),
    ]
    
    # Test data
    test_data = [
        ("The capital of Sweden is", " Stockholm"),
        ("The capital of Norway is", " Oslo"),
        ("The capital of Austria is", " Vienna"),
        ("The capital of Portugal is", " Lisbon"),
        ("The capital of Greece is", " Athens"),
    ]
    
    # Collect hidden states
    train_h = []
    train_answers = []
    
    for prompt, answer in train_data:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]
        
        train_h.append(h)
        
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        train_answers.append(embed[answer_ids[0]])
    
    H_train = torch.stack(train_h)
    A_train = torch.stack(train_answers)
    
    print(f"Training hidden states shape: {H_train.shape}")
    
    # Learn mapping: answer_embed = h @ W
    lambda_reg = 0.1
    HtH = H_train.T @ H_train + lambda_reg * torch.eye(H_train.shape[1])
    HtA = H_train.T @ A_train
    W = torch.linalg.solve(HtH, HtA)
    
    print(f"W shape: {W.shape}")
    
    # Test on training data
    print("\n--- Training Data ---")
    
    train_correct = 0
    for i, (prompt, answer) in enumerate(train_data):
        a_pred = H_train[i] @ W
        
        sims = F.cosine_similarity(a_pred.unsqueeze(0), embed)
        pred_idx = sims.argmax()
        pred_text = tokenizer.decode([pred_idx])
        
        marker = "✓" if pred_text.strip() == answer.strip() else "✗"
        if pred_text.strip() == answer.strip():
            train_correct += 1
        
        print(f"  {prompt[-20:]} → pred: {pred_text!r}, expected: {answer!r} {marker}")
    
    print(f"  Training accuracy: {train_correct}/{len(train_data)}")
    
    # Test on unseen data
    print("\n--- Unseen Data ---")
    
    test_correct = 0
    for prompt, expected in test_data:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]
            true_idx = outputs.logits[0, -1, :].argmax()
        
        true_text = tokenizer.decode([true_idx])
        
        a_pred = h @ W
        
        sims = F.cosine_similarity(a_pred.unsqueeze(0), embed)
        pred_idx = sims.argmax()
        pred_text = tokenizer.decode([pred_idx])
        
        marker = "✓" if pred_text.strip() == true_text.strip() else "✗"
        if pred_text.strip() == true_text.strip():
            test_correct += 1
        
        print(f"  {prompt[-20:]} → pred: {pred_text!r}, true: {true_text!r} {marker}")
    
    print(f"\n  Generalization accuracy: {test_correct}/{len(test_data)} = {test_correct/len(test_data)*100:.1f}%")


def main():
    print("=" * 70)
    print("Upcast Generalization V2")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Test 1: Direct linear mapping
    test_direct_linear_mapping(model, tokenizer)
    
    # Test 2: Offset approach
    test_offset_approach(model, tokenizer)
    
    # Test 3: Analogy approach
    test_analogy_approach(model, tokenizer)
    
    # Test 4: Hidden state approach
    test_hidden_state_approach(model, tokenizer)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    main()
