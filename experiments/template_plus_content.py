#!/usr/bin/env python3
"""
Template + Content: The Complete Model
=======================================

Previous finding:
- Template captures 98.8% of variance
- Residual (1.2%) contains the discriminative information
- Content adjustment is predictable (18.6% error)

The problem: Template alone predicts "______" (blank)
The solution: Template + Content adjustment

The model:
  h_final = template + W @ entity_embed
  output = argmax(h_final @ lm_head.T)

Where:
- template = mean hidden state for the pattern
- W = learned matrix mapping entity → residual
- entity_embed = embedding of the key entity in the input

This gives us PERFECT COVERAGE if:
1. We can detect the pattern (which template?)
2. We can extract the entity (which word?)
3. We can apply W (linear transformation)

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


def collect_data(model, tokenizer, prompts_with_entities):
    """
    Collect hidden states and entity embeddings.
    
    prompts_with_entities: list of (prompt, entity_word) tuples
    """
    embed = model.model.embed_tokens.weight.data
    results = []
    
    for prompt, entity_word in prompts_with_entities:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Get entity embedding
        entity_ids = tokenizer.encode(entity_word, add_special_tokens=False)
        if entity_ids:
            entity_embed = embed[entity_ids[0]]
        else:
            entity_embed = torch.zeros_like(h_final)
        
        results.append({
            'prompt': prompt,
            'entity_word': entity_word,
            'entity_embed': entity_embed,
            'h_final': h_final,
            'true_token': true_token,
            'true_text': tokenizer.decode([true_token]),
        })
    
    return results


def learn_template_and_W(results):
    """
    Learn the template and content adjustment matrix W.
    
    h_final = template + W @ entity_embed
    
    This is a linear regression problem.
    """
    H = torch.stack([r['h_final'] for r in results])
    E = torch.stack([r['entity_embed'] for r in results])
    
    # Template is the mean
    template = H.mean(dim=0)
    
    # Residuals
    R = H - template
    
    # Learn W such that R ≈ E @ W
    # W = (E^T E)^{-1} E^T R
    lambda_reg = 0.1
    EtE = E.T @ E + lambda_reg * torch.eye(E.shape[1])
    EtR = E.T @ R
    W = torch.linalg.solve(EtE, EtR)
    
    # Measure fit
    R_pred = E @ W
    error = (R - R_pred).norm() / R.norm()
    
    print(f"Template shape: {template.shape}")
    print(f"W shape: {W.shape}")
    print(f"Residual prediction error: {error*100:.1f}%")
    
    return template, W


def test_template_plus_content(template, W, model, tokenizer, test_data):
    """
    Test prediction using template + content adjustment.
    """
    print("\n" + "=" * 70)
    print("Template + Content Prediction Test")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    lm_head = model.lm_head.weight.data
    
    correct = 0
    
    for prompt, entity_word, expected in test_data:
        # Get entity embedding
        entity_ids = tokenizer.encode(entity_word, add_special_tokens=False)
        if entity_ids:
            entity_embed = embed[entity_ids[0]]
        else:
            entity_embed = torch.zeros(embed.shape[1])
        
        # Predict: h = template + W @ entity_embed
        # But W maps from entity_embed to residual, so:
        # h = template + entity_embed @ W
        h_pred = template + entity_embed @ W
        
        # Get logits
        logits = h_pred @ lm_head.T
        pred_token = logits.argmax().item()
        pred_text = tokenizer.decode([pred_token])
        
        # Get true answer
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0))
            true_token = outputs.logits[0, -1, :].argmax().item()
        true_text = tokenizer.decode([true_token])
        
        marker = "✓" if pred_token == true_token else "✗"
        if pred_token == true_token:
            correct += 1
        
        print(f"  {prompt!r} (entity: {entity_word!r})")
        print(f"    Pred: {pred_text!r}, True: {true_text!r} {marker}")
    
    print(f"\nAccuracy: {correct}/{len(test_data)} = {correct/len(test_data)*100:.1f}%")
    
    return correct / len(test_data)


def explore_pattern_specific_templates(model, tokenizer):
    """
    Learn separate templates for different patterns.
    
    Each pattern gets its own (template, W) pair.
    """
    print("\n" + "=" * 70)
    print("Pattern-Specific Templates")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    lm_head = model.lm_head.weight.data
    
    # Define patterns with training data
    patterns = {
        'capital': [
            ("The capital of France is", "France", " Paris"),
            ("The capital of Germany is", "Germany", " Berlin"),
            ("The capital of Italy is", "Italy", " Rome"),
            ("The capital of Spain is", "Spain", " Madrid"),
        ],
        'largest': [
            ("The largest planet is", "planet", " Jupiter"),
            ("The largest country is", "country", " Russia"),
            ("The largest ocean is", "ocean", " Pacific"),
        ],
        'math': [
            ("Two plus two equals", "Two", " four"),
            ("Three plus three equals", "Three", " six"),
            ("Five plus five equals", "Five", " ten"),
        ],
        'opposite': [
            ("The opposite of hot is", "hot", " cold"),
            ("The opposite of big is", "big", " small"),
            ("The opposite of fast is", "fast", " slow"),
        ],
    }
    
    # Learn template and W for each pattern
    pattern_models = {}
    
    for pattern_name, data in patterns.items():
        print(f"\n--- Pattern: {pattern_name} ---")
        
        prompts_with_entities = [(p, e) for p, e, _ in data]
        results = collect_data(model, tokenizer, prompts_with_entities)
        
        template, W = learn_template_and_W(results)
        
        pattern_models[pattern_name] = {
            'template': template,
            'W': W,
            'training_data': data,
        }
    
    # Test on training data
    print("\n" + "=" * 70)
    print("Testing on Training Data")
    print("=" * 70)
    
    total_correct = 0
    total_count = 0
    
    for pattern_name, model_data in pattern_models.items():
        print(f"\n--- Pattern: {pattern_name} ---")
        
        template = model_data['template']
        W = model_data['W']
        
        for prompt, entity, expected in model_data['training_data']:
            # Get entity embedding
            entity_ids = tokenizer.encode(entity, add_special_tokens=False)
            entity_embed = embed[entity_ids[0]]
            
            # Predict
            h_pred = template + entity_embed @ W
            logits = h_pred @ lm_head.T
            pred_token = logits.argmax().item()
            pred_text = tokenizer.decode([pred_token])
            
            # True answer
            expected_id = tokenizer.encode(expected, add_special_tokens=False)[0]
            
            marker = "✓" if pred_token == expected_id else "✗"
            if pred_token == expected_id:
                total_correct += 1
            total_count += 1
            
            print(f"  {prompt!r}: pred={pred_text!r}, expected={expected!r} {marker}")
    
    print(f"\nTotal training accuracy: {total_correct}/{total_count} = {total_correct/total_count*100:.1f}%")
    
    # Test on unseen data
    print("\n" + "=" * 70)
    print("Testing on Unseen Data")
    print("=" * 70)
    
    test_data = {
        'capital': [
            ("The capital of Japan is", "Japan", " Tokyo"),
            ("The capital of Poland is", "Poland", " Warsaw"),
            ("The capital of China is", "China", " Beijing"),
        ],
        'largest': [
            ("The largest continent is", "continent", " Asia"),
        ],
        'math': [
            ("One plus one equals", "One", " two"),
            ("Seven plus seven equals", "Seven", " fourteen"),
        ],
        'opposite': [
            ("The opposite of slow is", "slow", " fast"),
            ("The opposite of up is", "up", " down"),
        ],
    }
    
    total_correct = 0
    total_count = 0
    
    for pattern_name, test_cases in test_data.items():
        if pattern_name not in pattern_models:
            continue
        
        print(f"\n--- Pattern: {pattern_name} ---")
        
        template = pattern_models[pattern_name]['template']
        W = pattern_models[pattern_name]['W']
        
        for prompt, entity, expected in test_cases:
            # Get entity embedding
            entity_ids = tokenizer.encode(entity, add_special_tokens=False)
            entity_embed = embed[entity_ids[0]]
            
            # Predict
            h_pred = template + entity_embed @ W
            logits = h_pred @ lm_head.T
            pred_token = logits.argmax().item()
            pred_text = tokenizer.decode([pred_token])
            
            # True answer from transformer
            input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
            with torch.no_grad():
                outputs = model(input_ids.unsqueeze(0))
                true_token = outputs.logits[0, -1, :].argmax().item()
            true_text = tokenizer.decode([true_token])
            
            marker = "✓" if pred_token == true_token else "✗"
            if pred_token == true_token:
                total_correct += 1
            total_count += 1
            
            print(f"  {prompt!r}: pred={pred_text!r}, true={true_text!r} {marker}")
    
    print(f"\nTotal unseen accuracy: {total_correct}/{total_count} = {total_correct/total_count*100:.1f}%")
    
    return pattern_models


def analyze_what_W_encodes(pattern_models, model, tokenizer):
    """
    Analyze what W encodes.
    
    W maps entity embeddings to residuals.
    What is the structure of W?
    """
    print("\n" + "=" * 70)
    print("What Does W Encode?")
    print("=" * 70)
    
    for pattern_name, model_data in pattern_models.items():
        W = model_data['W']
        
        print(f"\n--- Pattern: {pattern_name} ---")
        print(f"  W shape: {W.shape}")
        print(f"  W norm: {W.norm():.2f}")
        
        # SVD of W
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        
        print(f"  Top 10 singular values: {S[:10].tolist()}")
        
        # Effective rank
        total_var = (S**2).sum()
        for k in [1, 5, 10, 37]:
            if k <= len(S):
                var_k = (S[:k]**2).sum() / total_var * 100
                print(f"  Top {k} components: {var_k:.1f}% variance")


def main():
    print("=" * 70)
    print("Template + Content: The Complete Model")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Learn pattern-specific templates
    pattern_models = explore_pattern_specific_templates(model, tokenizer)
    
    # Analyze W
    analyze_what_W_encodes(pattern_models, model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: What We're Converging To")
    print("=" * 70)
    print("""
The complete model is:

  h_final = template[pattern] + entity_embed @ W[pattern]
  output = argmax(h_final @ lm_head.T)

Where:
  - template[pattern]: Mean hidden state for the syntactic pattern
  - W[pattern]: Linear map from entity embedding to residual
  - entity_embed: Embedding of the key entity in the input

This is what we're converging to:
  1. A finite set of (template, W) pairs (one per pattern)
  2. A pattern detector (maps input → pattern)
  3. An entity extractor (maps input → key entity)

All three are GEOMETRIC operations on the DRUM/COMB structure.

The transformer's 28 layers are computing:
  - Pattern detection: Which template?
  - Entity extraction: Which word?
  - Linear combination: template + W @ entity

This is O(d²) instead of O(L × d² × N²).
""")


if __name__ == "__main__":
    main()
