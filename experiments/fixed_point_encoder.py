#!/usr/bin/env python3
"""
Fixed Point Encoder: Eliminating Hidden States
================================================

The hypothesis: If each token has a fixed point that predicts itself,
and context is just attention over previous fixed points, then we can
eliminate hidden states entirely.

Architecture:
    Input: [t₁, t₂, ..., tₙ] (token IDs)
    
    1. Lookup fixed points: [F[t₁], F[t₂], ..., F[tₙ]]
    2. Compute context via attention over fixed points
    3. Predict next: next_token = argmax(lm_head(context))
    
    Output: next_token

This is an ENCODER - tokens in, tokens out, no recurrence!

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


class FixedPointEncoder:
    """
    Encoder that uses pre-computed fixed points instead of hidden states.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_dim = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        
        # Extract lm_head for final projection
        self.lm_head = model.lm_head.weight.data.clone()  # [vocab_size, hidden_dim]
        
        # Fixed point table (to be populated)
        self.fixed_points = {}  # token_id -> fixed_point_vector
        
    def extract_fixed_points_from_generation(self, prompts: List[str], n_tokens: int = 20):
        """
        Extract fixed points by observing h_after for each token during generation.
        """
        print("Extracting fixed points from generation...")
        
        token_h_afters = defaultdict(list)
        
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            current_ids = input_ids.clone()
            
            for i in range(n_tokens):
                with torch.no_grad():
                    outputs = self.model(current_ids, output_hidden_states=True)
                    h_after = outputs.hidden_states[-1][0, -1, :].clone()
                    token = outputs.logits[0, -1, :].argmax().item()
                
                token_h_afters[token].append(h_after)
                current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
        
        # Compute mean fixed point for each token
        for token, h_afters in token_h_afters.items():
            if len(h_afters) >= 1:
                self.fixed_points[token] = torch.stack(h_afters).mean(dim=0)
        
        print(f"Extracted {len(self.fixed_points)} fixed points")
        return len(self.fixed_points)
    
    def get_fixed_point(self, token_id: int) -> torch.Tensor:
        """Get fixed point for a token, or use embedding as fallback."""
        if token_id in self.fixed_points:
            return self.fixed_points[token_id]
        else:
            # Fallback: use input embedding
            return self.model.model.embed_tokens.weight[token_id].clone()
    
    def encode_sequence(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence of tokens using fixed points.
        
        Returns: [seq_len, hidden_dim] tensor of fixed point representations
        """
        seq_len = token_ids.shape[-1]
        encoded = torch.zeros(seq_len, self.hidden_dim)
        
        for i in range(seq_len):
            token = token_ids[0, i].item()
            encoded[i] = self.get_fixed_point(token)
        
        return encoded
    
    def predict_next_simple(self, token_ids: torch.Tensor) -> int:
        """
        Simplest approach: use last token's fixed point directly.
        """
        last_token = token_ids[0, -1].item()
        fp = self.get_fixed_point(last_token)
        
        # Project through lm_head
        logits = fp @ self.lm_head.T
        return logits.argmax().item()
    
    def predict_next_mean(self, token_ids: torch.Tensor) -> int:
        """
        Use mean of all fixed points in sequence.
        """
        encoded = self.encode_sequence(token_ids)
        mean_fp = encoded.mean(dim=0)
        
        logits = mean_fp @ self.lm_head.T
        return logits.argmax().item()
    
    def predict_next_weighted(self, token_ids: torch.Tensor) -> int:
        """
        Use position-weighted sum of fixed points (more recent = higher weight).
        """
        encoded = self.encode_sequence(token_ids)
        seq_len = encoded.shape[0]
        
        # Exponential decay weights (more recent = higher)
        weights = torch.exp(torch.arange(seq_len).float() / seq_len)
        weights = weights / weights.sum()
        
        weighted_fp = (encoded * weights.unsqueeze(1)).sum(dim=0)
        
        logits = weighted_fp @ self.lm_head.T
        return logits.argmax().item()
    
    def predict_next_attention(self, token_ids: torch.Tensor) -> int:
        """
        Use self-attention over fixed points to compute context.
        """
        encoded = self.encode_sequence(token_ids)  # [seq_len, hidden_dim]
        
        # Simple dot-product attention
        # Query = last position, Keys/Values = all positions
        query = encoded[-1:]  # [1, hidden_dim]
        keys = encoded  # [seq_len, hidden_dim]
        values = encoded  # [seq_len, hidden_dim]
        
        # Attention scores
        scores = query @ keys.T / np.sqrt(self.hidden_dim)  # [1, seq_len]
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum
        context = attn_weights @ values  # [1, hidden_dim]
        
        logits = context @ self.lm_head.T
        return logits[0].argmax().item()
    
    def predict_next_hybrid(self, token_ids: torch.Tensor, alpha: float = 0.5) -> int:
        """
        Hybrid: combine last fixed point with attention context.
        
        h = alpha * F[last_token] + (1-alpha) * attention_context
        """
        encoded = self.encode_sequence(token_ids)
        
        # Last token's fixed point
        last_fp = encoded[-1]
        
        # Attention context (excluding last position to avoid self-attention)
        if encoded.shape[0] > 1:
            query = encoded[-1:]
            keys = encoded[:-1]
            values = encoded[:-1]
            
            scores = query @ keys.T / np.sqrt(self.hidden_dim)
            attn_weights = F.softmax(scores, dim=-1)
            context = (attn_weights @ values)[0]
        else:
            context = last_fp
        
        # Combine
        combined = alpha * last_fp + (1 - alpha) * context
        
        logits = combined @ self.lm_head.T
        return logits.argmax().item()


def test_fixed_point_encoder():
    """
    Test the fixed point encoder against the full model.
    """
    print("\n" + "=" * 70)
    print("Fixed Point Encoder Test")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Create encoder
    encoder = FixedPointEncoder(model, tokenizer)
    
    # Training prompts for fixed point extraction
    training_prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
        "Python is a programming language that",
        "The quick brown fox jumps over the",
        "To be or not to be that is the",
        "In the beginning there was",
        "Once upon a time there was a",
        "The meaning of life is",
        "Mathematics is the language of",
        "The sun rises in the",
        "Water is essential for",
        "Music is a form of",
        "The largest planet is",
    ]
    
    # Extract fixed points
    encoder.extract_fixed_points_from_generation(training_prompts, n_tokens=15)
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "The quick brown fox jumps over the",
        "Python is a programming language that",
        "The largest ocean is the",
        "The sun rises in the",
    ]
    
    print("\n" + "=" * 70)
    print("Testing Prediction Methods")
    print("=" * 70)
    
    methods = [
        ("Simple (last FP)", encoder.predict_next_simple),
        ("Mean FP", encoder.predict_next_mean),
        ("Weighted FP", encoder.predict_next_weighted),
        ("Attention FP", encoder.predict_next_attention),
        ("Hybrid (α=0.7)", lambda x: encoder.predict_next_hybrid(x, alpha=0.7)),
        ("Hybrid (α=0.5)", lambda x: encoder.predict_next_hybrid(x, alpha=0.5)),
        ("Hybrid (α=0.3)", lambda x: encoder.predict_next_hybrid(x, alpha=0.3)),
    ]
    
    results = {name: {"correct": 0, "total": 0} for name, _ in methods}
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        # Get ground truth from model
        with torch.no_grad():
            outputs = model(input_ids)
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        true_text = tokenizer.decode([true_token])
        
        print(f"\nPrompt: {prompt!r}")
        print(f"True next token: {true_text!r} (id={true_token})")
        
        for name, method in methods:
            pred_token = method(input_ids)
            pred_text = tokenizer.decode([pred_token])
            correct = pred_token == true_token
            
            results[name]["total"] += 1
            if correct:
                results[name]["correct"] += 1
            
            marker = "✓" if correct else "✗"
            print(f"  {name}: {pred_text!r} (id={pred_token}) {marker}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    for name, stats in results.items():
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {name}: {stats['correct']}/{stats['total']} = {acc:.1f}%")
    
    return encoder, results


def test_generation_comparison():
    """
    Compare full generation between model and encoder.
    """
    print("\n" + "=" * 70)
    print("Generation Comparison Test")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    encoder = FixedPointEncoder(model, tokenizer)
    
    # Extract fixed points with diverse prompts
    training_prompts = [
        "The capital of France is",
        "The capital of Germany is", 
        "The capital of Italy is",
        "The capital of Spain is",
        "The capital of Japan is",
        "The capital of China is",
        "Python is a programming language that",
        "Java is a programming language that",
        "The elephant is a large animal that",
        "The lion is a large animal that",
        "The quick brown fox jumps over the",
        "To be or not to be that is the",
        "In the beginning there was",
        "Once upon a time there was a",
        "The meaning of life is",
        "Mathematics is the language of",
        "The sun rises in the",
        "Water is essential for",
        "The largest planet is",
        "The smallest country is",
    ]
    
    encoder.extract_fixed_points_from_generation(training_prompts, n_tokens=20)
    
    # Test generation
    test_prompt = "The capital of France is"
    n_generate = 10
    
    print(f"\nPrompt: {test_prompt!r}")
    print(f"Generating {n_generate} tokens...\n")
    
    # Model generation
    input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
    model_tokens = []
    current_ids = input_ids.clone()
    
    for i in range(n_generate):
        with torch.no_grad():
            outputs = model(current_ids)
            token = outputs.logits[0, -1, :].argmax().item()
        model_tokens.append(token)
        current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
    
    model_text = tokenizer.decode(model_tokens)
    print(f"Model:   {test_prompt}{model_text}")
    
    # Encoder generation (various methods)
    methods = [
        ("Attention", encoder.predict_next_attention),
        ("Hybrid 0.7", lambda x: encoder.predict_next_hybrid(x, alpha=0.7)),
        ("Hybrid 0.5", lambda x: encoder.predict_next_hybrid(x, alpha=0.5)),
    ]
    
    for method_name, method in methods:
        encoder_tokens = []
        current_ids = input_ids.clone()
        
        for i in range(n_generate):
            token = method(current_ids)
            encoder_tokens.append(token)
            current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
        
        encoder_text = tokenizer.decode(encoder_tokens)
        
        # Count matching tokens
        matches = sum(1 for m, e in zip(model_tokens, encoder_tokens) if m == e)
        
        print(f"{method_name}: {test_prompt}{encoder_text} ({matches}/{n_generate} match)")
    
    # Detailed token-by-token comparison
    print("\n--- Token-by-Token Comparison ---")
    print(f"{'Pos':<4} {'Model':<15} {'Attention':<15} {'Match':<6}")
    print("-" * 45)
    
    # Re-run attention method for comparison
    encoder_tokens = []
    current_ids = input_ids.clone()
    for i in range(n_generate):
        token = encoder.predict_next_attention(current_ids)
        encoder_tokens.append(token)
        current_ids = torch.cat([current_ids, torch.tensor([[token]])], dim=1)
    
    for i, (m, e) in enumerate(zip(model_tokens, encoder_tokens)):
        m_text = tokenizer.decode([m])
        e_text = tokenizer.decode([e])
        match = "✓" if m == e else "✗"
        print(f"{i:<4} {m_text!r:<15} {e_text!r:<15} {match:<6}")
    
    return encoder


def analyze_what_context_encodes():
    """
    Analyze what information the context adjustment actually encodes.
    """
    print("\n" + "=" * 70)
    print("Context Analysis: What Does Context Encode?")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Compare same token in different contexts
    contexts = [
        ("The capital of France is", " Paris"),
        ("The capital of Germany is", " Berlin"),
        ("The capital of Italy is", " Rome"),
        ("The largest city in France is", " Paris"),
        ("A famous city in France is", " Paris"),
    ]
    
    print("\n--- Same Token, Different Contexts ---")
    
    token_data = defaultdict(list)
    
    for prompt, expected in contexts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            h_before = outputs.hidden_states[-1][0, -1, :].clone()
            
            # Generate one token
            token = outputs.logits[0, -1, :].argmax().item()
            token_text = tokenizer.decode([token])
            
            # Get h_after
            new_ids = torch.cat([input_ids, torch.tensor([[token]])], dim=1)
            outputs2 = model(new_ids, output_hidden_states=True)
            h_after = outputs2.hidden_states[-1][0, -1, :].clone()
        
        token_data[token_text].append({
            "prompt": prompt,
            "h_before": h_before,
            "h_after": h_after,
            "delta": h_after - h_before,
        })
        
        print(f"  {prompt!r} → {token_text!r}")
    
    # Analyze variance in h_after vs h_before for same token
    print("\n--- Consistency Analysis ---")
    
    for token_text, data_list in token_data.items():
        if len(data_list) < 2:
            continue
        
        h_befores = torch.stack([d["h_before"] for d in data_list])
        h_afters = torch.stack([d["h_after"] for d in data_list])
        deltas = torch.stack([d["delta"] for d in data_list])
        
        # Compute pairwise similarities
        h_before_norm = F.normalize(h_befores, dim=1)
        h_after_norm = F.normalize(h_afters, dim=1)
        delta_norm = F.normalize(deltas, dim=1)
        
        before_sim = (h_before_norm @ h_before_norm.T).mean().item()
        after_sim = (h_after_norm @ h_after_norm.T).mean().item()
        delta_sim = (delta_norm @ delta_norm.T).mean().item()
        
        print(f"\n  Token {token_text!r} (n={len(data_list)}):")
        print(f"    h_before similarity: {before_sim:.4f}")
        print(f"    h_after similarity:  {after_sim:.4f}")
        print(f"    delta similarity:    {delta_sim:.4f}")
        
        # What's the context adjustment?
        mean_h_after = h_afters.mean(dim=0)
        adjustments = h_afters - mean_h_after
        adj_norms = torch.norm(adjustments, dim=1)
        
        print(f"    Context adjustment norms: {adj_norms.tolist()}")
        print(f"    Mean adjustment: {adj_norms.mean().item():.2f}")
        print(f"    Mean h_after norm: {torch.norm(mean_h_after).item():.2f}")
        print(f"    Adjustment ratio: {(adj_norms.mean() / torch.norm(mean_h_after)).item()*100:.1f}%")


if __name__ == "__main__":
    # Test 1: Basic prediction methods
    encoder, results = test_fixed_point_encoder()
    
    # Test 2: Full generation comparison
    test_generation_comparison()
    
    # Test 3: Analyze what context encodes
    analyze_what_context_encodes()
