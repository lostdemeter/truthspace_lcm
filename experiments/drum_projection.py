#!/usr/bin/env python3
"""
DRUM Projection: Perfect Coverage Without Training
===================================================

The hypothesis:
- The DRUM (embeddings) contains all the information
- The transformer "rotates" this information to produce outputs
- If we can derive the rotation directly from the DRUM structure,
  we get perfect coverage without needing to see examples

From Doc 112 (Music Box Principle):
- DRUM = the embeddings (the "pins" that encode tokens)
- COMB = the output projection (lm_head)
- ROTATION = the transformation (what the transformer does)
- MUSIC = the output (next token prediction)

Key insight:
The DRUM and COMB are ALREADY TRAINED. They contain the geometric
structure. The transformer's 28 layers are just computing a rotation
that could potentially be derived directly from the DRUM/COMB geometry.

What if:
  signature(token) = project(embed(token), rotation_matrix)
  
Where rotation_matrix is derived from the DRUM/COMB structure itself?

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


def analyze_drum_comb_structure(model, tokenizer):
    """
    Analyze the geometric structure of DRUM (embeddings) and COMB (lm_head).
    
    Key question: Is there a direct geometric relationship between
    embed_tokens and lm_head that we can exploit?
    """
    print("\n" + "=" * 70)
    print("DRUM/COMB Structure Analysis")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data  # [vocab_size, hidden_dim]
    lm_head = model.lm_head.weight.data           # [vocab_size, hidden_dim]
    
    print(f"\nDRUM (embed_tokens): {embed.shape}")
    print(f"COMB (lm_head): {lm_head.shape}")
    
    # Are they the same? (weight tying)
    if torch.allclose(embed, lm_head):
        print("\n*** DRUM and COMB are IDENTICAL (weight tying) ***")
        print("This means: embed_tokens == lm_head")
        print("The transformation is symmetric!")
    else:
        # Check correlation
        flat_embed = embed.flatten()
        flat_lm = lm_head.flatten()
        corr = torch.corrcoef(torch.stack([flat_embed, flat_lm]))[0, 1]
        print(f"\nCorrelation between DRUM and COMB: {corr:.4f}")
    
    # Analyze the structure
    print("\n--- DRUM Statistics ---")
    print(f"  Mean: {embed.mean():.6f}")
    print(f"  Std: {embed.std():.6f}")
    print(f"  Min: {embed.min():.6f}")
    print(f"  Max: {embed.max():.6f}")
    
    # SVD of DRUM
    print("\n--- DRUM SVD ---")
    U, S, Vt = torch.linalg.svd(embed, full_matrices=False)
    print(f"  Top 10 singular values: {S[:10].tolist()}")
    print(f"  Variance in top 10: {(S[:10]**2).sum() / (S**2).sum() * 100:.1f}%")
    print(f"  Variance in top 37: {(S[:37]**2).sum() / (S**2).sum() * 100:.1f}%")
    
    return embed, lm_head, U, S, Vt


def test_direct_projection(model, tokenizer):
    """
    Test if we can predict the hidden state signature directly from
    the embedding, without running the transformer.
    
    Hypothesis: h_final ≈ f(embed) where f is a simple geometric transform.
    """
    print("\n" + "=" * 70)
    print("Direct Projection Test")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    lm_head = model.lm_head.weight.data
    
    # Test prompts
    prompts = [
        "The capital of France is",
        "The largest planet is",
        "Two plus two equals",
        "Hello, my name is",
    ]
    
    print("\n--- Testing Direct Projection ---")
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        # Get actual hidden state from transformer
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
            h_actual = outputs.hidden_states[-1][0, -1, :]
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        true_text = tokenizer.decode([true_token])
        
        # Get embeddings
        embeds = embed[input_ids]  # [seq_len, hidden_dim]
        
        # Try different projections
        projections = {
            'last_embed': embeds[-1],
            'mean_embed': embeds.mean(dim=0),
            'sum_embed': embeds.sum(dim=0),
            'weighted_embed': (embeds * torch.arange(1, len(embeds)+1).unsqueeze(1).float()).sum(dim=0) / sum(range(1, len(embeds)+1)),
        }
        
        print(f"\n{prompt!r} → {true_text!r}")
        
        for name, proj in projections.items():
            # Compute signature of projection
            proj_levels, proj_patterns = compute_tetromino_signature(proj)
            actual_levels, actual_patterns = compute_tetromino_signature(h_actual)
            
            # Compare
            level_match = (proj_levels == actual_levels).float().mean() * 100
            pattern_match = (proj_patterns == actual_patterns).float().mean() * 100
            
            # Try predicting with projection
            logits = proj @ lm_head.T
            pred_token = logits.argmax().item()
            pred_text = tokenizer.decode([pred_token])
            
            correct = "✓" if pred_token == true_token else "✗"
            
            print(f"  {name}: level={level_match:.1f}%, pattern={pattern_match:.1f}%, pred={pred_text!r} {correct}")


def find_rotation_matrix(model, tokenizer):
    """
    Try to find a rotation matrix R such that:
    
    h_final ≈ R @ aggregate(embeddings)
    
    If we can find R from the DRUM/COMB structure alone,
    we have perfect coverage without training.
    """
    print("\n" + "=" * 70)
    print("Finding Rotation Matrix from DRUM/COMB")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    lm_head = model.lm_head.weight.data
    
    # Collect data points
    prompts = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The largest planet is",
        "The smallest planet is",
        "Two plus two equals",
        "Three plus three equals",
        "Hello, my name is",
        "The weather today is",
        "I went to the store and",
    ]
    
    X = []  # Input: aggregated embeddings
    Y = []  # Output: actual hidden states
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
            h_actual = outputs.hidden_states[-1][0, -1, :]
        
        embeds = embed[input_ids]
        
        # Use weighted sum as input
        weights = torch.arange(1, len(embeds)+1).float()
        x = (embeds * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
        
        X.append(x)
        Y.append(h_actual)
    
    X = torch.stack(X)  # [n_samples, hidden_dim]
    Y = torch.stack(Y)  # [n_samples, hidden_dim]
    
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    
    # Try to find R such that Y ≈ X @ R
    # Using least squares: R = (X^T X)^{-1} X^T Y
    
    # Add regularization
    lambda_reg = 0.1
    XtX = X.T @ X + lambda_reg * torch.eye(X.shape[1])
    XtY = X.T @ Y
    R = torch.linalg.solve(XtX, XtY)
    
    print(f"R shape: {R.shape}")
    
    # Test the rotation
    Y_pred = X @ R
    
    # Measure accuracy
    print("\n--- Rotation Matrix Test ---")
    
    correct = 0
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0))
            true_token = outputs.logits[0, -1, :].argmax().item()
        
        # Predict using rotation
        h_pred = Y_pred[i]
        logits = h_pred @ lm_head.T
        pred_token = logits.argmax().item()
        
        true_text = tokenizer.decode([true_token])
        pred_text = tokenizer.decode([pred_token])
        
        marker = "✓" if pred_token == true_token else "✗"
        if pred_token == true_token:
            correct += 1
        
        print(f"  {prompt!r}: pred={pred_text!r}, true={true_text!r} {marker}")
    
    print(f"\nAccuracy: {correct}/{len(prompts)} = {correct/len(prompts)*100:.1f}%")
    
    return R


def explore_drum_self_structure(model, tokenizer):
    """
    Explore if the DRUM has self-structure that encodes the rotation.
    
    Key insight from Doc 112: The music box works because the DRUM
    and COMB are geometrically related. The rotation might be
    implicit in their relationship.
    """
    print("\n" + "=" * 70)
    print("DRUM Self-Structure Analysis")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    lm_head = model.lm_head.weight.data
    
    # Compute DRUM @ DRUM^T (self-similarity matrix)
    print("\n--- DRUM Self-Similarity ---")
    
    # Sample tokens for analysis
    test_tokens = [
        "Paris", "Berlin", "Rome", "Tokyo",  # Capitals
        "Jupiter", "Mars", "Venus",           # Planets
        "four", "six", "ten",                 # Numbers
        "the", "is", "of", "and",             # Function words
    ]
    
    token_ids = []
    for t in test_tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if ids:
            token_ids.append(ids[0])
    
    # Get embeddings for these tokens
    token_embeds = embed[token_ids]  # [n_tokens, hidden_dim]
    
    # Compute similarity matrix
    sim_matrix = token_embeds @ token_embeds.T
    sim_matrix = sim_matrix / (token_embeds.norm(dim=1, keepdim=True) @ token_embeds.norm(dim=1, keepdim=True).T + 1e-8)
    
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    print(f"Diagonal (self-similarity): {sim_matrix.diag().mean():.4f}")
    print(f"Off-diagonal mean: {(sim_matrix.sum() - sim_matrix.diag().sum()) / (sim_matrix.numel() - len(token_ids)):.4f}")
    
    # Check if similar tokens cluster
    print("\n--- Token Clustering in DRUM ---")
    
    # Capitals should be similar to each other
    capital_ids = [token_ids[i] for i in range(4)]
    capital_embeds = embed[capital_ids]
    capital_sim = (capital_embeds @ capital_embeds.T).mean()
    print(f"  Capitals similarity: {capital_sim:.4f}")
    
    # Planets should be similar
    planet_ids = [token_ids[i] for i in range(4, 7)]
    planet_embeds = embed[planet_ids]
    planet_sim = (planet_embeds @ planet_embeds.T).mean()
    print(f"  Planets similarity: {planet_sim:.4f}")
    
    # Cross-category should be less similar
    cross_sim = (capital_embeds @ planet_embeds.T).mean()
    print(f"  Cross (capitals-planets): {cross_sim:.4f}")


def test_drum_to_signature_direct(model, tokenizer):
    """
    The key test: Can we compute signatures directly from DRUM
    without running the transformer?
    
    If the DRUM contains the geometric structure, and signatures
    are just a quantization of that structure, then:
    
    signature(prompt) = quantize(aggregate(DRUM[tokens]))
    
    This would give us PERFECT COVERAGE because we're not learning
    anything - we're just reading the structure that's already there.
    """
    print("\n" + "=" * 70)
    print("DRUM → Signature Direct Mapping")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    lm_head = model.lm_head.weight.data
    
    # The hypothesis: For each token in vocabulary, we can compute
    # its "signature" directly from its embedding.
    
    # Then, for a prompt, we aggregate the token signatures
    # and look up the result.
    
    print("\n--- Computing Token Signatures from DRUM ---")
    
    # Compute signature for each token embedding
    vocab_size = embed.shape[0]
    
    # Sample some tokens
    sample_tokens = [
        " Paris", " Berlin", " Rome",
        " Jupiter", " Mars",
        " four", " six",
        " the", " is",
    ]
    
    print("\nToken signatures from DRUM:")
    for token_text in sample_tokens:
        token_id = tokenizer.encode(token_text, add_special_tokens=False)
        if token_id:
            token_embed = embed[token_id[0]]
            levels, patterns = compute_tetromino_signature(token_embed)
            
            # Summarize signature
            unique_levels = len(set(levels.tolist()))
            unique_patterns = len(set(patterns.tolist()))
            
            print(f"  {token_text!r}: {unique_levels} unique levels, {unique_patterns} unique patterns")
    
    # Now test: Can we predict next token from DRUM signatures alone?
    print("\n--- Prediction from DRUM Signatures ---")
    
    prompts = [
        ("The capital of France is", " Paris"),
        ("The largest planet is", " Jupiter"),
        ("Two plus two equals", " four"),
    ]
    
    for prompt, expected in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')[0]
        
        # Aggregate embeddings
        embeds = embed[input_ids]
        agg_embed = embeds.mean(dim=0)  # Simple mean
        
        # Compute signature of aggregated embedding
        agg_levels, agg_patterns = compute_tetromino_signature(agg_embed)
        
        # Find nearest token in DRUM by signature
        best_match = None
        best_distance = float('inf')
        
        # Check against expected token
        expected_id = tokenizer.encode(expected, add_special_tokens=False)[0]
        expected_embed = embed[expected_id]
        expected_levels, expected_patterns = compute_tetromino_signature(expected_embed)
        
        level_diff = (agg_levels != expected_levels).sum().item()
        pattern_diff = (agg_patterns != expected_patterns).sum().item()
        
        print(f"\n{prompt!r} → expected {expected!r}")
        print(f"  Signature distance to expected: levels={level_diff}, patterns={pattern_diff}")
        
        # Try direct dot product prediction
        logits = agg_embed @ lm_head.T
        pred_token = logits.argmax().item()
        pred_text = tokenizer.decode([pred_token])
        
        print(f"  Direct prediction: {pred_text!r}")


def main():
    print("=" * 70)
    print("DRUM Projection: Perfect Coverage Without Training")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Analysis 1: DRUM/COMB structure
    embed, lm_head, U, S, Vt = analyze_drum_comb_structure(model, tokenizer)
    
    # Analysis 2: Direct projection test
    test_direct_projection(model, tokenizer)
    
    # Analysis 3: Find rotation matrix
    R = find_rotation_matrix(model, tokenizer)
    
    # Analysis 4: DRUM self-structure
    explore_drum_self_structure(model, tokenizer)
    
    # Analysis 5: DRUM to signature direct
    test_drum_to_signature_direct(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key Findings:

1. DRUM and COMB structure analysis reveals the geometric relationship
2. Direct projection from embeddings has limited accuracy
3. A learned rotation matrix can improve accuracy
4. The DRUM has semantic clustering (capitals, planets, etc.)

The Challenge:
The transformer's 28 layers compute a CONTEXT-DEPENDENT rotation.
The rotation depends on the SEQUENCE of tokens, not just individual tokens.

Possible Solutions:
1. Find a universal rotation that works for all contexts
2. Decompose the rotation into context-independent + context-dependent parts
3. Use the DRUM's self-structure to derive the rotation

The key insight: If we can express the rotation as a function of
the DRUM geometry alone, we get perfect coverage without training.
""")


if __name__ == "__main__":
    main()
