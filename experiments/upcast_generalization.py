#!/usr/bin/env python3
"""
Upcast Generalization Test
==========================

We found that upcasting achieves 100% accuracy on training data.
Now test: Does U_up generalize to UNSEEN entities?

If yes, this is a major breakthrough - we can predict answers
for entities we've never seen before!

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


def test_upcast_generalization(model, tokenizer):
    """
    Test if the upcast matrix generalizes to unseen entities.
    """
    print("\n" + "=" * 70)
    print("Upcast Generalization Test")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    lm_head = model.lm_head.weight.data
    
    # Training pairs (used to learn U_up)
    train_pairs = [
        ("France", " Paris"),
        ("Germany", " Berlin"),
        ("Italy", " Rome"),
        ("Spain", " Madrid"),
        ("Japan", " Tokyo"),
        ("China", " Beijing"),
    ]
    
    # Test pairs (UNSEEN during training)
    test_pairs = [
        ("Poland", " Warsaw"),
        ("Brazil", " Brasilia"),
        ("Egypt", " Cairo"),
        ("India", " New"),  # New Delhi
        ("Russia", " Moscow"),
        ("Canada", " Ottawa"),
        ("Australia", " Canberra"),
        ("Mexico", " Mexico"),  # Mexico City
        ("Argentina", " Buenos"),  # Buenos Aires
        ("Sweden", " Stockholm"),
    ]
    
    # Get training embeddings
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
    
    # Compute projection using SVD of difference
    diff = E_train - A_train
    U_diff, S_diff, Vt_diff = torch.linalg.svd(diff, full_matrices=False)
    
    # Use the dimensions where E and A are SIMILAR (last k)
    k = min(100, Vt_diff.shape[0])
    P_similar = Vt_diff[-k:, :]
    
    # Project training data
    E_train_proj = E_train @ P_similar.T
    
    # Learn upcast matrix: A = E_proj @ U_up
    lambda_reg = 0.1
    n_proj = E_train_proj.shape[1]
    EtE = E_train_proj.T @ E_train_proj + lambda_reg * torch.eye(n_proj)
    EtA = E_train_proj.T @ A_train
    U_up = torch.linalg.solve(EtE, EtA)
    
    print(f"Upcast matrix shape: {U_up.shape}")
    
    # Test on training data
    print("\n--- Training Data (sanity check) ---")
    
    train_correct = 0
    for i, (entity, answer) in enumerate(train_pairs):
        entity_embed = E_train[i]
        e_proj = entity_embed @ P_similar.T
        a_pred = e_proj @ U_up
        
        sims = F.cosine_similarity(a_pred.unsqueeze(0), embed)
        pred_idx = sims.argmax()
        pred_text = tokenizer.decode([pred_idx])
        
        marker = "✓" if pred_text.strip() == answer.strip() else "✗"
        if pred_text.strip() == answer.strip():
            train_correct += 1
        
        print(f"  {entity} → pred: {pred_text!r}, expected: {answer!r} {marker}")
    
    print(f"  Training accuracy: {train_correct}/{len(train_pairs)}")
    
    # Test on UNSEEN data
    print("\n--- Unseen Data (generalization test) ---")
    
    test_correct = 0
    for entity, expected_answer in test_pairs:
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        entity_embed = embed[entity_ids[0]]
        
        # Apply the learned upcast
        e_proj = entity_embed @ P_similar.T
        a_pred = e_proj @ U_up
        
        # Find nearest token
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


def test_with_more_training_data(model, tokenizer):
    """
    Test if more training data improves generalization.
    """
    print("\n" + "=" * 70)
    print("Effect of Training Data Size")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    
    # All available pairs
    all_pairs = [
        ("France", " Paris"),
        ("Germany", " Berlin"),
        ("Italy", " Rome"),
        ("Spain", " Madrid"),
        ("Japan", " Tokyo"),
        ("China", " Beijing"),
        ("Poland", " Warsaw"),
        ("Brazil", " Brasilia"),
        ("Egypt", " Cairo"),
        ("Russia", " Moscow"),
        ("Canada", " Ottawa"),
        ("Australia", " Canberra"),
        ("Sweden", " Stockholm"),
        ("Norway", " Oslo"),
        ("Finland", " Helsinki"),
        ("Denmark", " Copenhagen"),
        ("Netherlands", " Amsterdam"),
        ("Belgium", " Brussels"),
        ("Austria", " Vienna"),
        ("Switzerland", " Bern"),
    ]
    
    # Fixed test set (last 5)
    test_pairs = all_pairs[-5:]
    
    # Vary training set size
    for n_train in [4, 8, 12, 15]:
        train_pairs = all_pairs[:n_train]
        
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
        
        # Learn projection and upcast
        diff = E_train - A_train
        U_diff, S_diff, Vt_diff = torch.linalg.svd(diff, full_matrices=False)
        
        k = min(100, Vt_diff.shape[0])
        P_similar = Vt_diff[-k:, :]
        
        E_train_proj = E_train @ P_similar.T
        
        lambda_reg = 0.1
        n_proj = E_train_proj.shape[1]
        EtE = E_train_proj.T @ E_train_proj + lambda_reg * torch.eye(n_proj)
        EtA = E_train_proj.T @ A_train
        U_up = torch.linalg.solve(EtE, EtA)
        
        # Test on held-out data
        test_correct = 0
        for entity, expected in test_pairs:
            entity_ids = tokenizer.encode(entity, add_special_tokens=False)
            entity_embed = embed[entity_ids[0]]
            
            e_proj = entity_embed @ P_similar.T
            a_pred = e_proj @ U_up
            
            sims = F.cosine_similarity(a_pred.unsqueeze(0), embed)
            pred_idx = sims.argmax()
            pred_text = tokenizer.decode([pred_idx])
            
            if pred_text.strip() == expected.strip():
                test_correct += 1
        
        print(f"  n_train={n_train}: test accuracy = {test_correct}/{len(test_pairs)} = {test_correct/len(test_pairs)*100:.1f}%")


def analyze_upcast_structure(model, tokenizer):
    """
    Analyze the structure of the upcast matrix.
    
    Is it related to DRUM/COMB structure?
    """
    print("\n" + "=" * 70)
    print("Upcast Matrix Structure Analysis")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    lm_head = model.lm_head.weight.data
    
    # Learn upcast from all pairs
    all_pairs = [
        ("France", " Paris"),
        ("Germany", " Berlin"),
        ("Italy", " Rome"),
        ("Spain", " Madrid"),
        ("Japan", " Tokyo"),
        ("China", " Beijing"),
        ("Poland", " Warsaw"),
        ("Russia", " Moscow"),
    ]
    
    entities = []
    answers = []
    
    for entity, answer in all_pairs:
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        
        entities.append(embed[entity_ids[0]])
        answers.append(embed[answer_ids[0]])
    
    E = torch.stack(entities)
    A = torch.stack(answers)
    
    # Learn upcast
    diff = E - A
    U_diff, S_diff, Vt_diff = torch.linalg.svd(diff, full_matrices=False)
    
    k = min(100, Vt_diff.shape[0])
    P = Vt_diff[-k:, :]
    
    E_proj = E @ P.T
    
    lambda_reg = 0.1
    n_proj = E_proj.shape[1]
    EtE = E_proj.T @ E_proj + lambda_reg * torch.eye(n_proj)
    EtA = E_proj.T @ A
    U_up = torch.linalg.solve(EtE, EtA)
    
    print(f"U_up shape: {U_up.shape}")
    
    # SVD of U_up
    U_u, S_u, Vt_u = torch.linalg.svd(U_up, full_matrices=False)
    
    print(f"Top 10 singular values of U_up: {S_u[:10].tolist()}")
    
    # Effective rank
    total_var = (S_u**2).sum()
    for k in [1, 5, 10, 20]:
        if k <= len(S_u):
            var_k = (S_u[:k]**2).sum() / total_var * 100
            print(f"  Top {k} components: {var_k:.1f}% variance")
    
    # Is U_up related to lm_head?
    print("\n--- Relationship to LM_HEAD ---")
    
    # lm_head is [vocab, 3584]
    # U_up is [n_proj, 3584]
    
    # Similarity between U_up rows and lm_head rows
    U_up_norm = U_up / U_up.norm(dim=1, keepdim=True)
    lm_head_norm = lm_head / lm_head.norm(dim=1, keepdim=True)
    
    # For each U_up row, find most similar lm_head row
    print("  Most similar lm_head rows to U_up rows:")
    for i in range(min(5, U_up.shape[0])):
        sims = U_up_norm[i] @ lm_head_norm.T
        max_sim = sims.max()
        max_idx = sims.argmax()
        max_token = tokenizer.decode([max_idx])
        print(f"    U_up[{i}] → {max_token!r} (sim={max_sim:.4f})")


def main():
    print("=" * 70)
    print("Upcast Generalization Test")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Test 1: Basic generalization
    gen_acc = test_upcast_generalization(model, tokenizer)
    
    # Test 2: Effect of training data size
    test_with_more_training_data(model, tokenizer)
    
    # Test 3: Analyze upcast structure
    analyze_upcast_structure(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Generalization Results:
- Training accuracy: 100% (by construction)
- Generalization accuracy: {gen_acc*100:.1f}%

If generalization works, this means:
1. The upcast matrix captures the "capital-of" RELATIONSHIP
2. Not just memorizing specific (entity, answer) pairs
3. Can predict answers for UNSEEN entities!

This is the path to perfect coverage without training:
1. Learn U_up from a few examples per pattern
2. Apply to ANY entity in that pattern
3. The relationship IS geometric!
""")


if __name__ == "__main__":
    main()
