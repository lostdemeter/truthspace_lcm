#!/usr/bin/env python3
"""
Sign-Based Encoder: Using σ=0.5 Sign Structure
===============================================

Key insights from Docs 165/166:
1. Sign patterns encode semantics (not magnitudes)
2. σ=0.5 (60 dims) is optimal LOD
3. 50% universal + 50% dimension-specific structure

The problem with previous tests:
- Using raw embeddings/hidden states gives garbage
- Need to work in SIGN SPACE at the critical line

New approach:
1. Project to σ=0.5 (60 dims via SVD)
2. Work with SIGNS only
3. Use sign-based matching for prediction

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

PHI = 1.6180339887498949


class SignBasedEncoder:
    """
    Encoder that works in sign space at σ=0.5.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_dim = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        
        # Compute σ=0.5 dimension
        self.k = int(np.sqrt(self.hidden_dim))  # ~60 for 3584
        print(f"σ=0.5 dimension: {self.k}")
        
        # Get embeddings and lm_head
        self.embed = model.model.embed_tokens.weight.data.clone()
        self.lm_head = model.lm_head.weight.data.clone()
        
        # Compute SVD projection to σ=0.5
        print("Computing SVD projection...")
        U, S, Vt = torch.linalg.svd(self.embed, full_matrices=False)
        self.projection = Vt[:self.k, :].T  # [hidden_dim, k]
        
        # Project embeddings to low-dim
        self.embed_low = self.embed @ self.projection  # [vocab, k]
        self.embed_signs = torch.sign(self.embed_low)  # [vocab, k]
        
        # Project lm_head to low-dim
        self.lm_head_low = self.lm_head @ self.projection  # [vocab, k]
        self.lm_head_signs = torch.sign(self.lm_head_low)
        
        print(f"Embed low shape: {self.embed_low.shape}")
        print(f"LM head low shape: {self.lm_head_low.shape}")
        
        # Collect hidden state projections
        self.hidden_projections = {}
    
    def project_hidden(self, h: torch.Tensor) -> torch.Tensor:
        """Project hidden state to σ=0.5."""
        return h @ self.projection
    
    def get_hidden_signs(self, h: torch.Tensor) -> torch.Tensor:
        """Get sign pattern of hidden state at σ=0.5."""
        return torch.sign(self.project_hidden(h))
    
    def predict_from_signs(self, h_signs: torch.Tensor) -> int:
        """Predict next token using sign agreement."""
        # Agreement with each vocabulary embedding
        agreement = (self.embed_signs == h_signs).sum(dim=1).float()
        return agreement.argmax().item()
    
    def predict_from_low(self, h_low: torch.Tensor) -> int:
        """Predict next token using low-dim cosine similarity."""
        sims = F.cosine_similarity(h_low.unsqueeze(0), self.embed_low, dim=1)
        return sims.argmax().item()
    
    def predict_from_lm_head_signs(self, h_signs: torch.Tensor) -> int:
        """Predict using sign agreement with lm_head."""
        agreement = (self.lm_head_signs == h_signs).sum(dim=1).float()
        return agreement.argmax().item()
    
    def predict_from_lm_head_low(self, h_low: torch.Tensor) -> int:
        """Predict using low-dim dot product with lm_head."""
        logits = h_low @ self.lm_head_low.T
        return logits.argmax().item()


def test_sign_based_prediction():
    """
    Test sign-based prediction at σ=0.5.
    """
    print("\n" + "=" * 70)
    print("Sign-Based Prediction at σ=0.5")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    encoder = SignBasedEncoder(model, tokenizer)
    
    # Test prompts
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The largest ocean is the",
        "Python is a programming language that",
        "The quick brown fox jumps over the",
    ]
    
    print("\n" + "=" * 70)
    print("Test 1: Direct hidden state → sign-based prediction")
    print("=" * 70)
    
    results = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_true = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        true_text = tokenizer.decode([true_token])
        
        # Project to σ=0.5
        h_low = encoder.project_hidden(h_true)
        h_signs = torch.sign(h_low)
        
        print(f"\nPrompt: {prompt!r}")
        print(f"True: {true_text!r} (id={true_token})")
        
        # Method 1: Sign agreement with embeddings
        pred1 = encoder.predict_from_signs(h_signs)
        pred1_text = tokenizer.decode([pred1])
        m1 = "✓" if pred1 == true_token else "✗"
        results["sign_embed"]["total"] += 1
        if pred1 == true_token:
            results["sign_embed"]["correct"] += 1
        print(f"  Sign→Embed: {pred1_text!r} {m1}")
        
        # Method 2: Low-dim cosine with embeddings
        pred2 = encoder.predict_from_low(h_low)
        pred2_text = tokenizer.decode([pred2])
        m2 = "✓" if pred2 == true_token else "✗"
        results["low_embed"]["total"] += 1
        if pred2 == true_token:
            results["low_embed"]["correct"] += 1
        print(f"  Low→Embed:  {pred2_text!r} {m2}")
        
        # Method 3: Sign agreement with lm_head
        pred3 = encoder.predict_from_lm_head_signs(h_signs)
        pred3_text = tokenizer.decode([pred3])
        m3 = "✓" if pred3 == true_token else "✗"
        results["sign_lm"]["total"] += 1
        if pred3 == true_token:
            results["sign_lm"]["correct"] += 1
        print(f"  Sign→LM:    {pred3_text!r} {m3}")
        
        # Method 4: Low-dim dot with lm_head
        pred4 = encoder.predict_from_lm_head_low(h_low)
        pred4_text = tokenizer.decode([pred4])
        m4 = "✓" if pred4 == true_token else "✗"
        results["low_lm"]["total"] += 1
        if pred4 == true_token:
            results["low_lm"]["correct"] += 1
        print(f"  Low→LM:     {pred4_text!r} {m4}")
        
        # Method 5: Full-dim lm_head (baseline)
        pred5 = (h_true @ encoder.lm_head.T).argmax().item()
        pred5_text = tokenizer.decode([pred5])
        m5 = "✓" if pred5 == true_token else "✗"
        results["full_lm"]["total"] += 1
        if pred5 == true_token:
            results["full_lm"]["correct"] += 1
        print(f"  Full→LM:    {pred5_text!r} {m5}")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    for name, stats in results.items():
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {name}: {stats['correct']}/{stats['total']} = {acc:.1f}%")
    
    return encoder, results


def test_sign_structure_of_hidden_states():
    """
    Analyze the sign structure of hidden states.
    """
    print("\n" + "=" * 70)
    print("Analyzing Sign Structure of Hidden States")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    encoder = SignBasedEncoder(model, tokenizer)
    
    # Collect hidden states for different prompts
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "Python is a programming language that",
        "Java is a programming language that",
    ]
    
    hidden_states = []
    next_tokens = []
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]
            token = outputs.logits[0, -1, :].argmax().item()
        
        hidden_states.append(h)
        next_tokens.append(token)
    
    hidden_states = torch.stack(hidden_states)
    
    # Project to σ=0.5
    h_low = hidden_states @ encoder.projection
    h_signs = torch.sign(h_low)
    
    print(f"\nHidden states shape: {hidden_states.shape}")
    print(f"Low-dim shape: {h_low.shape}")
    
    # Analyze sign patterns
    print("\n--- Sign Pattern Analysis ---")
    
    # How many dimensions have consistent signs across prompts?
    sign_consistency = (h_signs == h_signs[0]).float().mean(dim=0)
    consistent_dims = (sign_consistency > 0.8).sum().item()
    print(f"Dimensions with >80% sign consistency: {consistent_dims}/{encoder.k}")
    
    # Sign similarity between hidden states
    print("\n--- Sign Similarity Matrix ---")
    for i in range(len(prompts)):
        for j in range(len(prompts)):
            sim = (h_signs[i] == h_signs[j]).float().mean().item()
            print(f"  {i}-{j}: {sim:.3f}", end="")
        print()
    
    # Compare hidden state signs to next token embedding signs
    print("\n--- Hidden → Next Token Sign Agreement ---")
    
    for i, (prompt, token) in enumerate(zip(prompts, next_tokens)):
        h_sign = h_signs[i]
        token_embed_sign = encoder.embed_signs[token]
        
        agreement = (h_sign == token_embed_sign).float().mean().item()
        token_text = tokenizer.decode([token])
        
        print(f"  {prompt!r} → {token_text!r}: {agreement*100:.1f}% sign agreement")
    
    # What's the sign agreement with the CORRECT token vs random tokens?
    print("\n--- Sign Agreement: Correct vs Random ---")
    
    for i, (prompt, token) in enumerate(zip(prompts, next_tokens)):
        h_sign = h_signs[i]
        
        # Agreement with correct token
        correct_agree = (h_sign == encoder.embed_signs[token]).float().mean().item()
        
        # Agreement with random tokens
        random_tokens = torch.randint(0, encoder.vocab_size, (100,))
        random_agrees = [(h_sign == encoder.embed_signs[t]).float().mean().item() 
                         for t in random_tokens]
        random_mean = np.mean(random_agrees)
        random_std = np.std(random_agrees)
        
        token_text = tokenizer.decode([token])
        z_score = (correct_agree - random_mean) / random_std if random_std > 0 else 0
        
        print(f"  {token_text!r}: correct={correct_agree:.3f}, random={random_mean:.3f}±{random_std:.3f}, z={z_score:.2f}")
    
    return encoder


def test_sequence_to_hidden_sign_mapping():
    """
    Can we predict hidden state SIGNS from input token SIGNS?
    """
    print("\n" + "=" * 70)
    print("Sequence Signs → Hidden Signs Mapping")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    encoder = SignBasedEncoder(model, tokenizer)
    
    # Training data
    training_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "Python is a programming language that",
        "Java is a programming language that",
        "The quick brown fox jumps over the",
        "To be or not to be that is the",
        "The largest planet is",
        "The smallest country is",
        "Water is essential for",
        "The sun rises in the",
    ]
    
    # Collect (input_signs, output_signs) pairs
    X_signs = []  # Input: XOR of token signs
    Y_signs = []  # Output: hidden state signs
    Y_tokens = []  # True next tokens
    
    for prompt in training_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]
            token = outputs.logits[0, -1, :].argmax().item()
        
        # Input: combine token signs (try XOR, OR, majority vote)
        token_signs = encoder.embed_signs[input_ids[0]]  # [seq_len, k]
        
        # Method 1: Last token signs
        input_sign_last = token_signs[-1]
        
        # Method 2: XOR of all signs
        input_sign_xor = token_signs[0].clone()
        for i in range(1, len(token_signs)):
            input_sign_xor = input_sign_xor * token_signs[i]  # XOR in {-1,+1}
        
        # Method 3: Majority vote
        input_sign_maj = torch.sign(token_signs.sum(dim=0))
        
        # Output: hidden state signs
        h_low = encoder.project_hidden(h)
        h_sign = torch.sign(h_low)
        
        X_signs.append({
            "last": input_sign_last,
            "xor": input_sign_xor,
            "maj": input_sign_maj,
        })
        Y_signs.append(h_sign)
        Y_tokens.append(token)
    
    Y_signs = torch.stack(Y_signs)
    
    # Test each input method
    print("\n--- Input Sign Methods ---")
    
    for method in ["last", "xor", "maj"]:
        X = torch.stack([x[method] for x in X_signs])
        
        # How well do input signs predict output signs?
        agreement = (X == Y_signs).float().mean(dim=1)
        
        print(f"\n{method.upper()} method:")
        print(f"  Mean sign agreement: {agreement.mean().item()*100:.1f}%")
        print(f"  Min: {agreement.min().item()*100:.1f}%")
        print(f"  Max: {agreement.max().item()*100:.1f}%")
        
        # Can we use input signs directly to predict next token?
        correct = 0
        for i, (x_sign, true_token) in enumerate(zip(X, Y_tokens)):
            pred = encoder.predict_from_signs(x_sign)
            if pred == true_token:
                correct += 1
        
        print(f"  Direct prediction accuracy: {correct}/{len(Y_tokens)} = {correct/len(Y_tokens)*100:.1f}%")
    
    # Learn a sign flip pattern: which dimensions flip from input to output?
    print("\n--- Learning Sign Flip Pattern ---")
    
    X_last = torch.stack([x["last"] for x in X_signs])
    
    # Flip pattern: where do signs differ between input and output?
    flip_pattern = (X_last != Y_signs).float().mean(dim=0)
    
    print(f"Mean flip rate per dimension: {flip_pattern.mean().item()*100:.1f}%")
    print(f"Dimensions that always flip: {(flip_pattern > 0.9).sum().item()}")
    print(f"Dimensions that never flip: {(flip_pattern < 0.1).sum().item()}")
    
    # Apply learned flip pattern
    flip_mask = flip_pattern > 0.5  # Dimensions that flip more than half the time
    
    correct = 0
    for i, (x_sign, true_token) in enumerate(zip(X_last, Y_tokens)):
        # Apply flip
        pred_sign = x_sign.clone()
        pred_sign[flip_mask] *= -1
        
        pred = encoder.predict_from_signs(pred_sign)
        if pred == true_token:
            correct += 1
    
    print(f"\nWith learned flip pattern: {correct}/{len(Y_tokens)} = {correct/len(Y_tokens)*100:.1f}%")
    
    return encoder


if __name__ == "__main__":
    # Test 1: Sign-based prediction from true hidden states
    encoder, results = test_sign_based_prediction()
    
    # Test 2: Analyze sign structure
    test_sign_structure_of_hidden_states()
    
    # Test 3: Sequence signs → hidden signs
    test_sequence_to_hidden_sign_mapping()
