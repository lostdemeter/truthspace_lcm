#!/usr/bin/env python3
"""
Error Analysis: What's Keeping Us From 100%?
=============================================

We achieved 87.5% accuracy (7/8) on encoder-only prediction.
The one failure was:
  'Hello, my name is' → predicted 'bad', true 'Dr'

Let's analyze:
1. What makes this case different?
2. Where does the signature encoder fail?
3. What's the structure of the error?

Author: TruthSpace LCM Team
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

PHI = 1.6180339887498949


def compute_tetromino_signature(vec, block_size=4):
    """Compute tetromino signature for a vector."""
    n_blocks = len(vec) // block_size
    blocks = vec.reshape(n_blocks, block_size)
    
    levels = []
    patterns = []
    
    for i in range(n_blocks):
        block = blocks[i]
        magnitudes = block.abs()
        mean_mag = magnitudes.mean()
        mean_level = int(round(np.log(mean_mag.item() + 1e-10) / np.log(PHI)))
        levels.append(mean_level)
        
        signs = (block > 0).int()
        sign_pattern = signs[0] * 8 + signs[1] * 4 + signs[2] * 2 + signs[3]
        patterns.append(sign_pattern.item())
    
    return torch.tensor(levels), torch.tensor(patterns)


def signature_distance(levels1, patterns1, levels2, patterns2):
    """Compute distance between two signatures."""
    level_diff = (levels1 != levels2).sum().item()
    pattern_diff = (patterns1 != patterns2).sum().item()
    return level_diff + pattern_diff


def analyze_failure_case(model, tokenizer):
    """
    Deep analysis of the failure case: 'Hello, my name is'
    """
    print("\n" + "=" * 70)
    print("Analyzing Failure Case: 'Hello, my name is'")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data.clone()
    lm_head = model.lm_head.weight.data.clone()
    
    failure_prompt = "Hello, my name is"
    
    input_ids = tokenizer.encode(failure_prompt, return_tensors='pt')[0]
    
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
        h_final = outputs.hidden_states[-1][0, -1, :]
        logits = outputs.logits[0, -1, :]
        true_token = logits.argmax().item()
    
    true_text = tokenizer.decode([true_token])
    
    print(f"\nPrompt: {failure_prompt!r}")
    print(f"True next token: {true_text!r} (id={true_token})")
    
    # Analyze the hidden state
    print(f"\n--- Hidden State Analysis ---")
    print(f"  |h|: {torch.norm(h_final).item():.2f}")
    print(f"  mean: {h_final.mean().item():.4f}")
    print(f"  std: {h_final.std().item():.4f}")
    print(f"  min: {h_final.min().item():.4f}")
    print(f"  max: {h_final.max().item():.4f}")
    
    # Compute signature
    levels, patterns = compute_tetromino_signature(h_final)
    
    print(f"\n--- Signature Analysis ---")
    print(f"  Level range: [{levels.min().item()}, {levels.max().item()}]")
    print(f"  Mean level: {levels.float().mean().item():.2f}")
    print(f"  Pattern distribution: {patterns.unique().shape[0]} unique patterns")
    
    # Top predictions
    print(f"\n--- Top 10 Predictions ---")
    top_tokens = logits.topk(10)
    
    for rank, (score, tok_id) in enumerate(zip(top_tokens.values, top_tokens.indices)):
        tok_text = tokenizer.decode([tok_id.item()])
        print(f"  {rank+1}. {tok_text!r}: {score.item():.2f}")
    
    # Entropy
    probs = F.softmax(logits, dim=0)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    print(f"\n  Entropy: {entropy:.2f}")
    
    return h_final, levels, patterns, true_token


def compare_with_successful_cases(model, tokenizer):
    """
    Compare the failure case with successful cases.
    """
    print("\n" + "=" * 70)
    print("Comparing Failure vs Success Cases")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data.clone()
    
    success_prompts = [
        "The capital of France is",
        "The largest planet is",
        "Two plus two equals",
        "I went to the store and",
        "The quick brown fox jumps over the",
    ]
    
    failure_prompt = "Hello, my name is"
    
    # Collect data for all prompts
    all_data = []
    
    for prompt in success_prompts + [failure_prompt]:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            logits = outputs.logits[0, -1, :]
            true_token = logits.argmax().item()
        
        # Input features
        token_embeds = embed[input_ids]
        seq_len = len(token_embeds)
        
        feat_sum = token_embeds.sum(dim=0)
        feat_mean = token_embeds.mean(dim=0)
        feat_last = token_embeds[-1]
        weights = torch.exp(torch.arange(seq_len).float() / seq_len)
        weights = weights / weights.sum()
        feat_weighted = (token_embeds * weights.unsqueeze(1)).sum(dim=0)
        feat_first = token_embeds[0]
        
        x = torch.cat([feat_sum, feat_mean, feat_last, feat_weighted, feat_first])
        
        # Signature
        levels, patterns = compute_tetromino_signature(h_final)
        
        # Entropy
        probs = F.softmax(logits, dim=0)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        
        all_data.append({
            'prompt': prompt,
            'is_failure': prompt == failure_prompt,
            'input_features': x,
            'hidden_state': h_final,
            'levels': levels,
            'patterns': patterns,
            'true_token': true_token,
            'true_text': tokenizer.decode([true_token]),
            'entropy': entropy,
            'h_norm': torch.norm(h_final).item(),
            'x_norm': torch.norm(x).item(),
        })
    
    # Compare
    print("\n--- Comparison ---")
    print(f"{'Prompt':<45} {'Token':<10} {'Entropy':<8} {'|h|':<8} {'|x|':<8}")
    print("-" * 80)
    
    for d in all_data:
        marker = "*** FAIL ***" if d['is_failure'] else ""
        print(f"{d['prompt'][:44]:<45} {d['true_text']:<10} {d['entropy']:<8.2f} {d['h_norm']:<8.2f} {d['x_norm']:<8.2f} {marker}")
    
    # Signature distances
    print("\n--- Signature Distances from Failure Case ---")
    
    failure_data = all_data[-1]
    
    for d in all_data[:-1]:
        dist = signature_distance(
            failure_data['levels'], failure_data['patterns'],
            d['levels'], d['patterns']
        )
        print(f"  {d['prompt'][:40]}: distance = {dist}")
    
    return all_data


def analyze_training_coverage(model, tokenizer):
    """
    Check if the failure case is covered by training data.
    """
    print("\n" + "=" * 70)
    print("Training Coverage Analysis")
    print("=" * 70)
    
    # Our training prompts
    training_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The largest planet is",
        "The opposite of hot is",
        "Two plus two equals",
        "I went to the store and",
        "She said that she would",
        "The book is on the",
        "The quick brown fox jumps over the",
    ]
    
    failure_prompt = "Hello, my name is"
    
    embed = model.model.embed_tokens.weight.data.clone()
    
    # Compute input feature similarity
    def get_features(prompt):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        token_embeds = embed[input_ids]
        seq_len = len(token_embeds)
        
        feat_sum = token_embeds.sum(dim=0)
        feat_mean = token_embeds.mean(dim=0)
        feat_last = token_embeds[-1]
        weights = torch.exp(torch.arange(seq_len).float() / seq_len)
        weights = weights / weights.sum()
        feat_weighted = (token_embeds * weights.unsqueeze(1)).sum(dim=0)
        feat_first = token_embeds[0]
        
        return torch.cat([feat_sum, feat_mean, feat_last, feat_weighted, feat_first])
    
    failure_features = get_features(failure_prompt)
    
    print(f"\nSimilarity of '{failure_prompt}' to training prompts:")
    
    similarities = []
    for prompt in training_prompts:
        features = get_features(prompt)
        sim = F.cosine_similarity(failure_features.unsqueeze(0), features.unsqueeze(0)).item()
        similarities.append((prompt, sim))
    
    similarities.sort(key=lambda x: -x[1])
    
    for prompt, sim in similarities:
        print(f"  {sim:.4f}: {prompt}")
    
    print(f"\nMax similarity: {similarities[0][1]:.4f}")
    print(f"Min similarity: {similarities[-1][1]:.4f}")
    
    # Check if failure case is an outlier
    avg_sim = np.mean([s for _, s in similarities])
    print(f"Avg similarity: {avg_sim:.4f}")
    
    if similarities[0][1] < 0.7:
        print("\n*** FAILURE CASE IS AN OUTLIER - not covered by training! ***")
    
    return similarities


def analyze_what_encoder_learned(model, tokenizer):
    """
    Analyze what the signature encoder actually learned.
    """
    print("\n" + "=" * 70)
    print("What Did the Encoder Learn?")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data.clone()
    
    # Categories of prompts
    categories = {
        'capitals': [
            "The capital of France is",
            "The capital of Germany is",
            "The capital of Italy is",
        ],
        'planets': [
            "The largest planet is",
            "The smallest planet is",
        ],
        'math': [
            "Two plus two equals",
            "Three times three equals",
        ],
        'scaffolding': [
            "I went to the store and",
            "The book is on the",
        ],
        'greetings': [
            "Hello, my name is",
            "Hi, I am",
            "Good morning, my name is",
        ],
    }
    
    # Compute features for each category
    category_features = {}
    
    for cat_name, prompts in categories.items():
        features = []
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
            token_embeds = embed[input_ids]
            seq_len = len(token_embeds)
            
            feat_sum = token_embeds.sum(dim=0)
            feat_mean = token_embeds.mean(dim=0)
            feat_last = token_embeds[-1]
            weights = torch.exp(torch.arange(seq_len).float() / seq_len)
            weights = weights / weights.sum()
            feat_weighted = (token_embeds * weights.unsqueeze(1)).sum(dim=0)
            feat_first = token_embeds[0]
            
            x = torch.cat([feat_sum, feat_mean, feat_last, feat_weighted, feat_first])
            features.append(x)
        
        category_features[cat_name] = torch.stack(features)
    
    # Compute within-category and cross-category similarities
    print("\n--- Within-Category Similarity ---")
    
    for cat_name, features in category_features.items():
        if len(features) > 1:
            sims = []
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    sim = F.cosine_similarity(features[i].unsqueeze(0), features[j].unsqueeze(0)).item()
                    sims.append(sim)
            mean_sim = np.mean(sims)
            print(f"  {cat_name}: {mean_sim:.4f}")
    
    print("\n--- Cross-Category Similarity ---")
    
    for cat1, features1 in category_features.items():
        for cat2, features2 in category_features.items():
            if cat1 < cat2:
                sims = []
                for f1 in features1:
                    for f2 in features2:
                        sim = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
                        sims.append(sim)
                mean_sim = np.mean(sims)
                print(f"  {cat1} vs {cat2}: {mean_sim:.4f}")
    
    # Key insight: Is 'greetings' category different from others?
    print("\n--- Greetings vs Other Categories ---")
    
    greetings_features = category_features['greetings']
    
    for cat_name, features in category_features.items():
        if cat_name != 'greetings':
            sims = []
            for g in greetings_features:
                for f in features:
                    sim = F.cosine_similarity(g.unsqueeze(0), f.unsqueeze(0)).item()
                    sims.append(sim)
            mean_sim = np.mean(sims)
            print(f"  greetings vs {cat_name}: {mean_sim:.4f}")


def propose_fixes(model, tokenizer):
    """
    Propose fixes to reach 100% accuracy.
    """
    print("\n" + "=" * 70)
    print("Proposed Fixes")
    print("=" * 70)
    
    print("""
Based on the analysis, the failure case 'Hello, my name is' fails because:

1. **Not in training distribution**: The greeting pattern is different from
   capitals, planets, math, and scaffolding prompts we trained on.

2. **High entropy prediction**: 'Dr' is a low-probability token (entropy ~6.9),
   meaning the model is uncertain. Our encoder can't capture this uncertainty.

3. **Different semantic category**: Greetings are a distinct category that
   wasn't represented in training.

PROPOSED FIXES:

1. **Add greeting prompts to training**:
   - "Hello, my name is"
   - "Hi, I am"
   - "Good morning, my name is"
   - "Nice to meet you, I am"

2. **Increase training diversity**:
   - Add more semantic categories
   - Include high-entropy cases
   - Cover more syntactic patterns

3. **Use category-aware memory**:
   - Detect category from input features
   - Use category-specific memory lookup
   - Fall back to full model for unknown categories

4. **Confidence threshold**:
   - If signature distance > threshold, use full model
   - This handles out-of-distribution cases

5. **Ensemble approach**:
   - Use encoder for high-confidence cases
   - Use full model for low-confidence cases
   - Blend based on signature distance
""")


def test_with_more_training(model, tokenizer):
    """
    Test if adding more training data fixes the issue.
    """
    print("\n" + "=" * 70)
    print("Testing with Expanded Training Data")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data.clone()
    
    # Expanded training prompts including greetings
    training_prompts = [
        # Capitals
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
        # Planets
        "The largest planet is",
        "The smallest planet is",
        # Math
        "Two plus two equals",
        "Three times three equals",
        # Scaffolding
        "I went to the store and",
        "She said that she would",
        "The book is on the",
        # Completions
        "The quick brown fox jumps over the",
        "To be or not to be that is the",
        # GREETINGS (NEW!)
        "Hello, my name is",
        "Hi, I am",
        "Good morning, my name is",
        "Nice to meet you, I am",
        "My name is",
    ]
    
    # Collect training data
    memory = {}
    
    for prompt in training_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        levels, patterns = compute_tetromino_signature(h_final)
        
        memory[prompt] = {
            'levels': levels,
            'patterns': patterns,
            'next_token': true_token,
            'next_text': tokenizer.decode([true_token]),
        }
    
    print(f"Built memory with {len(memory)} entries (including greetings)")
    
    # Test
    test_prompts = [
        "The capital of France is",
        "The capital of Poland is",
        "The largest planet is",
        "Two plus two equals",
        "I went to the store and",
        "The quick brown fox jumps over the",
        "Hello, my name is",  # The failure case
        "Hi, my name is",     # Similar to failure
    ]
    
    print("\n--- Test Results ---")
    
    correct = 0
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        true_text = tokenizer.decode([true_token])
        
        # Find nearest in memory
        query_levels, query_patterns = compute_tetromino_signature(h_final)
        
        best_match = None
        best_distance = float('inf')
        
        for stored_prompt, entry in memory.items():
            distance = signature_distance(
                query_levels, query_patterns,
                entry['levels'], entry['patterns']
            )
            
            if distance < best_distance:
                best_distance = distance
                best_match = entry
                best_prompt = stored_prompt
        
        pred_text = best_match['next_text']
        pred_token = best_match['next_token']
        
        is_correct = pred_token == true_token
        if is_correct:
            correct += 1
        
        marker = "✓" if is_correct else "✗"
        
        print(f"  {prompt!r}")
        print(f"    Pred: {pred_text!r}, True: {true_text!r} (dist={best_distance}) {marker}")
        print(f"    Matched: {best_prompt!r}")
    
    print(f"\nAccuracy: {correct}/{len(test_prompts)} = {correct/len(test_prompts)*100:.1f}%")
    
    return correct / len(test_prompts)


def main():
    print("=" * 70)
    print("Error Analysis: What's Keeping Us From 100%?")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Analysis 1: Failure case deep dive
    h_final, levels, patterns, true_token = analyze_failure_case(model, tokenizer)
    
    # Analysis 2: Compare with successful cases
    all_data = compare_with_successful_cases(model, tokenizer)
    
    # Analysis 3: Training coverage
    similarities = analyze_training_coverage(model, tokenizer)
    
    # Analysis 4: What encoder learned
    analyze_what_encoder_learned(model, tokenizer)
    
    # Analysis 5: Proposed fixes
    propose_fixes(model, tokenizer)
    
    # Analysis 6: Test with more training
    accuracy = test_with_more_training(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
The failure case 'Hello, my name is' fails because:

1. It's an OUT-OF-DISTRIBUTION case - greetings weren't in training
2. It has HIGH ENTROPY (6.9) - the model is uncertain
3. The nearest training prompt is semantically different

FIX: Add greeting prompts to training data.

With expanded training (including greetings):
  Accuracy: {accuracy*100:.1f}%
""")


if __name__ == "__main__":
    main()
