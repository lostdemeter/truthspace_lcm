#!/usr/bin/env python3
"""
Content Token Intersection: Finding Paris at the Intersection
==============================================================

Hypothesis: Content tokens like "Paris" are found at the INTERSECTION
of multiple semantic constraints:

  Paris = France ∩ Capital ∩ City ∩ European

The model learned to navigate to this intersection.

From Doc 039: φ-Zipf ordering determines importance.
From Doc 164: Signs encode semantics.

New approach:
1. Get embeddings for constraint words (France, capital, city, European)
2. Find the intersection point
3. See if Paris is near that intersection

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2


def get_embedding(embeddings, tokenizer, word: str) -> torch.Tensor:
    """Get embedding for a word (first token if multi-token)."""
    ids = tokenizer.encode(word, add_special_tokens=False)
    if ids:
        return embeddings[ids[0]]
    return None


def find_intersection_point(embeddings: List[torch.Tensor], method: str = "mean") -> torch.Tensor:
    """
    Find the intersection point of multiple embeddings.
    
    Methods:
    - mean: Simple average
    - weighted: φ-weighted average (rare words weighted more)
    - projection: Project onto shared subspace
    """
    if method == "mean":
        return torch.stack(embeddings).mean(dim=0)
    elif method == "weighted":
        # Weight by inverse magnitude (rarer = smaller magnitude = higher weight)
        mags = torch.tensor([e.norm().item() for e in embeddings])
        weights = 1.0 / mags
        weights = weights / weights.sum()
        weighted = sum(w * e for w, e in zip(weights, embeddings))
        return weighted
    else:
        return torch.stack(embeddings).mean(dim=0)


def test_intersection_hypothesis(embeddings, tokenizer):
    """
    Test: Is Paris at the intersection of France + capital + city?
    """
    print("=" * 70)
    print("INTERSECTION HYPOTHESIS TEST")
    print("=" * 70)
    
    test_cases = [
        {
            "target": "Paris",
            "constraints": ["France", "capital", "city"],
        },
        {
            "target": "Berlin",
            "constraints": ["Germany", "capital", "city"],
        },
        {
            "target": "Tokyo",
            "constraints": ["Japan", "capital", "city"],
        },
        {
            "target": "physicist",
            "constraints": ["Einstein", "science", "profession"],
        },
    ]
    
    for case in test_cases:
        target = case["target"]
        constraints = case["constraints"]
        
        print(f"\n--- {target} = {' ∩ '.join(constraints)} ---")
        
        # Get embeddings
        target_emb = get_embedding(embeddings, tokenizer, target)
        constraint_embs = [get_embedding(embeddings, tokenizer, c) for c in constraints]
        
        if target_emb is None or any(e is None for e in constraint_embs):
            print("  Missing embeddings")
            continue
        
        # Find intersection point
        intersection = find_intersection_point(constraint_embs, method="mean")
        
        # Distance from intersection to target
        dist_to_target = (intersection - target_emb).norm().item()
        
        # Find nearest token to intersection
        distances = (embeddings - intersection.unsqueeze(0)).norm(dim=1)
        nearest_idx = distances.argmin().item()
        nearest_token = tokenizer.decode([nearest_idx]).strip()
        nearest_dist = distances[nearest_idx].item()
        
        # Top 10 nearest
        top10_indices = distances.argsort()[:10]
        top10_tokens = [tokenizer.decode([idx.item()]).strip() for idx in top10_indices]
        
        print(f"  Intersection → nearest: {repr(nearest_token)} (dist={nearest_dist:.4f})")
        print(f"  Intersection → target:  {repr(target)} (dist={dist_to_target:.4f})")
        print(f"  Top 10: {top10_tokens}")
        
        # Is target in top 10?
        target_rank = None
        for i, idx in enumerate(distances.argsort()):
            if tokenizer.decode([idx.item()]).strip().lower() == target.lower():
                target_rank = i + 1
                break
        
        if target_rank:
            print(f"  Target rank: {target_rank}")
        else:
            print(f"  Target not in top 1000")


def test_vector_arithmetic(embeddings, tokenizer):
    """
    Test classic word2vec-style arithmetic:
    France - country + capital ≈ Paris?
    """
    print("\n" + "=" * 70)
    print("VECTOR ARITHMETIC TEST")
    print("=" * 70)
    
    # king - man + woman = queen style tests
    tests = [
        ("France", "country", "capital", "Paris"),
        ("Germany", "country", "capital", "Berlin"),
        ("France", "France", "Germany", "Germany"),  # Sanity check
        ("king", "man", "woman", "queen"),
        ("Einstein", "scientist", "artist", "Picasso"),
    ]
    
    for a, b, c, expected in tests:
        a_emb = get_embedding(embeddings, tokenizer, a)
        b_emb = get_embedding(embeddings, tokenizer, b)
        c_emb = get_embedding(embeddings, tokenizer, c)
        
        if a_emb is None or b_emb is None or c_emb is None:
            print(f"\n{a} - {b} + {c} = ? (missing embeddings)")
            continue
        
        # a - b + c
        result = a_emb - b_emb + c_emb
        
        # Find nearest
        distances = (embeddings - result.unsqueeze(0)).norm(dim=1)
        nearest_idx = distances.argmin().item()
        nearest_token = tokenizer.decode([nearest_idx]).strip()
        
        top5_indices = distances.argsort()[:5]
        top5_tokens = [tokenizer.decode([idx.item()]).strip() for idx in top5_indices]
        
        is_correct = nearest_token.lower() == expected.lower()
        status = "✓" if is_correct else "✗"
        
        print(f"\n{a} - {b} + {c} = {nearest_token} (expected: {expected}) {status}")
        print(f"  Top 5: {top5_tokens}")


def test_hidden_state_intersection(model, tokenizer, device):
    """
    Test: Does the hidden state after "The capital of France is"
    lie at the intersection of France + capital + city in hidden space?
    """
    print("\n" + "=" * 70)
    print("HIDDEN STATE INTERSECTION TEST")
    print("=" * 70)
    
    # Get hidden state after prompt
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0, -1, :].float().cpu()
    
    print(f"\nPrompt: {repr(prompt)}")
    print(f"Hidden state shape: {hidden.shape}")
    
    # Get hidden states for individual words
    words = ["France", "capital", "city", "Paris"]
    word_hiddens = {}
    
    for word in words:
        word_prompt = f"The word {word} means"
        word_ids = tokenizer.encode(word_prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(word_ids, output_hidden_states=True)
            word_hidden = outputs.hidden_states[-1][0, -1, :].float().cpu()
        
        word_hiddens[word] = word_hidden
    
    # Compute distances
    print(f"\nDistances from prompt hidden state:")
    for word, h in word_hiddens.items():
        dist = (hidden - h).norm().item()
        cos_sim = torch.nn.functional.cosine_similarity(hidden.unsqueeze(0), h.unsqueeze(0)).item()
        print(f"  {word}: dist={dist:.2f}, cos_sim={cos_sim:.4f}")
    
    # Intersection of France + capital
    intersection = (word_hiddens["France"] + word_hiddens["capital"]) / 2
    dist_to_intersection = (hidden - intersection).norm().item()
    print(f"\n  Intersection(France, capital): dist={dist_to_intersection:.2f}")
    
    # Is hidden closer to Paris than to intersection?
    dist_to_paris = (hidden - word_hiddens["Paris"]).norm().item()
    print(f"  Paris: dist={dist_to_paris:.2f}")


def analyze_lm_head_geometry(model, tokenizer, device):
    """
    The LM head maps hidden states to vocabulary.
    Analyze its geometry to understand content token prediction.
    """
    print("\n" + "=" * 70)
    print("LM HEAD GEOMETRY ANALYSIS")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data.float().cpu()
    print(f"\nLM head shape: {lm_head.shape}")
    
    # Get hidden state after "The capital of France is"
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0, -1, :].float().cpu()
    
    # Compute logits
    logits = hidden @ lm_head.T
    
    # Top predictions
    top_indices = logits.argsort(descending=True)[:20]
    
    print(f"\nTop 20 predictions after '{prompt}':")
    for i, idx in enumerate(top_indices):
        token = tokenizer.decode([idx.item()])
        logit = logits[idx].item()
        print(f"  {i+1:2d}. {repr(token):15s} logit={logit:.2f}")
    
    # Where is Paris?
    paris_ids = tokenizer.encode("Paris", add_special_tokens=False)
    if paris_ids:
        paris_idx = paris_ids[0]
        paris_logit = logits[paris_idx].item()
        paris_rank = (logits > paris_logit).sum().item() + 1
        print(f"\n  Paris: rank={paris_rank}, logit={paris_logit:.2f}")
    
    # Analyze the geometry: what makes Paris rank high?
    # Compare Paris's lm_head row to the hidden state
    paris_row = lm_head[paris_idx]
    cos_sim = torch.nn.functional.cosine_similarity(hidden.unsqueeze(0), paris_row.unsqueeze(0)).item()
    print(f"  Hidden-Paris cosine similarity: {cos_sim:.4f}")


def main():
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    
    embeddings = model.model.embed_tokens.weight.data.float().cpu()
    
    # Test intersection hypothesis
    test_intersection_hypothesis(embeddings, tokenizer)
    
    # Test vector arithmetic
    test_vector_arithmetic(embeddings, tokenizer)
    
    # Test hidden state intersection
    test_hidden_state_intersection(model, tokenizer, device)
    
    # Analyze LM head geometry
    analyze_lm_head_geometry(model, tokenizer, device)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    main()
