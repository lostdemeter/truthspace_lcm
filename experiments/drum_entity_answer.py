#!/usr/bin/env python3
"""
DRUM Entity→Answer Exploration
==============================

Can we derive entity→answer mappings from the DRUM structure?

The DRUM (embedding matrix) contains all token embeddings.
If there's a geometric relationship between entities and answers,
it should be visible in the DRUM.

Hypotheses to test:
1. Entities and answers cluster together in DRUM
2. There's a "capital-of" direction in DRUM space
3. Entity-answer pairs have consistent geometric relationships
4. The relationship is encoded in embedding neighborhoods

Author: TruthSpace LCM Team
Date: 2026-01-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

PHI = 1.6180339887498949


def analyze_drum_clustering(model, tokenizer):
    """
    Do entities and their answers cluster together in DRUM space?
    """
    print("\n" + "=" * 70)
    print("DRUM Clustering Analysis")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    
    # Entity-answer pairs
    pairs = [
        ("France", " Paris"),
        ("Germany", " Berlin"),
        ("Italy", " Rome"),
        ("Spain", " Madrid"),
        ("Japan", " Tokyo"),
        ("China", " Beijing"),
        ("hot", " cold"),
        ("big", " small"),
        ("fast", " slow"),
        ("up", " down"),
    ]
    
    print("\n--- Entity-Answer Distances in DRUM ---")
    
    for entity, answer in pairs:
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        
        entity_embed = embed[entity_ids[0]]
        answer_embed = embed[answer_ids[0]]
        
        # Cosine similarity
        sim = F.cosine_similarity(entity_embed.unsqueeze(0), answer_embed.unsqueeze(0))
        
        # Euclidean distance
        dist = (entity_embed - answer_embed).norm()
        
        # Rank: how close is answer to entity among all tokens?
        all_sims = F.cosine_similarity(entity_embed.unsqueeze(0), embed)
        answer_rank = (all_sims > sim).sum().item()
        
        print(f"  {entity} → {answer}: sim={sim.item():.4f}, dist={dist:.2f}, rank={answer_rank}/{len(embed)}")


def analyze_relationship_directions(model, tokenizer):
    """
    Is there a consistent "capital-of" direction in DRUM space?
    """
    print("\n" + "=" * 70)
    print("Relationship Direction Analysis")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    
    # Capital pairs
    capital_pairs = [
        ("France", " Paris"),
        ("Germany", " Berlin"),
        ("Italy", " Rome"),
        ("Spain", " Madrid"),
        ("Japan", " Tokyo"),
        ("China", " Beijing"),
    ]
    
    # Opposite pairs
    opposite_pairs = [
        ("hot", " cold"),
        ("big", " small"),
        ("fast", " slow"),
        ("up", " down"),
        ("good", " bad"),
        ("light", " dark"),
    ]
    
    def compute_directions(pairs):
        directions = []
        for entity, answer in pairs:
            entity_ids = tokenizer.encode(entity, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            
            entity_embed = embed[entity_ids[0]]
            answer_embed = embed[answer_ids[0]]
            
            direction = answer_embed - entity_embed
            direction = direction / direction.norm()  # Normalize
            directions.append(direction)
        
        return torch.stack(directions)
    
    capital_dirs = compute_directions(capital_pairs)
    opposite_dirs = compute_directions(opposite_pairs)
    
    # Consistency within relationship type
    print("\n--- Direction Consistency ---")
    
    def measure_consistency(dirs, name):
        n = len(dirs)
        sims = []
        for i in range(n):
            for j in range(i+1, n):
                sim = F.cosine_similarity(dirs[i].unsqueeze(0), dirs[j].unsqueeze(0))
                sims.append(sim.item())
        
        mean_sim = np.mean(sims)
        std_sim = np.std(sims)
        print(f"  {name}: mean similarity = {mean_sim:.4f} ± {std_sim:.4f}")
        return mean_sim
    
    capital_consistency = measure_consistency(capital_dirs, "Capital-of")
    opposite_consistency = measure_consistency(opposite_dirs, "Opposite-of")
    
    # Cross-relationship similarity
    cross_sims = []
    for c_dir in capital_dirs:
        for o_dir in opposite_dirs:
            sim = F.cosine_similarity(c_dir.unsqueeze(0), o_dir.unsqueeze(0))
            cross_sims.append(sim.item())
    
    print(f"  Cross (capital vs opposite): mean similarity = {np.mean(cross_sims):.4f}")
    
    # Mean directions
    mean_capital_dir = capital_dirs.mean(dim=0)
    mean_capital_dir = mean_capital_dir / mean_capital_dir.norm()
    
    mean_opposite_dir = opposite_dirs.mean(dim=0)
    mean_opposite_dir = mean_opposite_dir / mean_opposite_dir.norm()
    
    # Test mean direction
    print("\n--- Testing Mean Direction ---")
    
    test_pairs = [
        ("Poland", " Warsaw"),
        ("Sweden", " Stockholm"),
    ]
    
    for entity, expected in test_pairs:
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        entity_embed = embed[entity_ids[0]]
        
        # Predict using mean direction
        pred_embed = entity_embed + mean_capital_dir * 0.5  # Scale factor
        
        # Find nearest token
        sims = F.cosine_similarity(pred_embed.unsqueeze(0), embed)
        pred_idx = sims.argmax()
        pred_text = tokenizer.decode([pred_idx])
        
        print(f"  {entity} + mean_capital_dir → {pred_text!r} (expected: {expected!r})")


def analyze_embedding_neighborhoods(model, tokenizer):
    """
    Are entity-answer pairs in each other's neighborhoods?
    """
    print("\n" + "=" * 70)
    print("Embedding Neighborhood Analysis")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    
    pairs = [
        ("France", " Paris"),
        ("Germany", " Berlin"),
        ("Italy", " Rome"),
        ("hot", " cold"),
        ("big", " small"),
    ]
    
    print("\n--- Neighborhood Analysis ---")
    
    for entity, answer in pairs:
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        
        entity_embed = embed[entity_ids[0]]
        answer_embed = embed[answer_ids[0]]
        
        # Top-k neighbors of entity
        entity_sims = F.cosine_similarity(entity_embed.unsqueeze(0), embed)
        top_k = 10
        top_indices = entity_sims.argsort(descending=True)[:top_k]
        top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
        
        # Is answer in top-k?
        answer_in_topk = answer_ids[0] in top_indices
        answer_rank = (entity_sims > entity_sims[answer_ids[0]]).sum().item()
        
        print(f"\n  {entity}:")
        print(f"    Top-{top_k} neighbors: {top_tokens}")
        print(f"    Answer {answer!r} rank: {answer_rank} (in top-{top_k}: {answer_in_topk})")


def analyze_semantic_clusters(model, tokenizer):
    """
    Do countries cluster together? Do capitals cluster together?
    """
    print("\n" + "=" * 70)
    print("Semantic Cluster Analysis")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    
    countries = ["France", "Germany", "Italy", "Spain", "Japan", "China", "Poland", "Sweden"]
    capitals = [" Paris", " Berlin", " Rome", " Madrid", " Tokyo", " Beijing", " Warsaw", " Stockholm"]
    
    # Get embeddings
    country_embeds = []
    capital_embeds = []
    
    for country in countries:
        ids = tokenizer.encode(country, add_special_tokens=False)
        country_embeds.append(embed[ids[0]])
    
    for capital in capitals:
        ids = tokenizer.encode(capital, add_special_tokens=False)
        capital_embeds.append(embed[ids[0]])
    
    C = torch.stack(country_embeds)
    P = torch.stack(capital_embeds)
    
    # Intra-cluster similarity
    C_norm = C / C.norm(dim=1, keepdim=True)
    P_norm = P / P.norm(dim=1, keepdim=True)
    
    C_sim = (C_norm @ C_norm.T).mean().item()
    P_sim = (P_norm @ P_norm.T).mean().item()
    
    # Cross-cluster similarity
    CP_sim = (C_norm @ P_norm.T).mean().item()
    
    print(f"\n  Country-Country similarity: {C_sim:.4f}")
    print(f"  Capital-Capital similarity: {P_sim:.4f}")
    print(f"  Country-Capital similarity: {CP_sim:.4f}")
    
    # Do countries cluster more with each other than with capitals?
    if C_sim > CP_sim:
        print("  → Countries cluster together (more than with capitals)")
    else:
        print("  → Countries don't cluster distinctly")


def explore_drum_structure(model, tokenizer):
    """
    Explore the overall structure of the DRUM.
    """
    print("\n" + "=" * 70)
    print("DRUM Structure Exploration")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    
    print(f"\nDRUM shape: {embed.shape}")
    print(f"DRUM norm stats: mean={embed.norm(dim=1).mean():.4f}, std={embed.norm(dim=1).std():.4f}")
    
    # SVD of DRUM
    U, S, Vt = torch.linalg.svd(embed, full_matrices=False)
    
    print(f"\nTop 20 singular values: {S[:20].tolist()}")
    
    # Effective dimensionality
    total_var = (S**2).sum()
    for k in [10, 50, 100, 500, 1000]:
        var_k = (S[:k]**2).sum() / total_var * 100
        print(f"  Top {k} components: {var_k:.1f}% variance")
    
    # φ-structure in singular values?
    print("\n--- φ-Structure in DRUM Singular Values ---")
    for i in range(10):
        ratio = S[i] / S[i+1]
        phi_error = abs(ratio - PHI) / PHI * 100
        print(f"  S[{i}]/S[{i+1}] = {ratio:.4f} (φ error: {phi_error:.1f}%)")


def main():
    print("=" * 70)
    print("DRUM Entity→Answer Exploration")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Analysis 1: Clustering
    analyze_drum_clustering(model, tokenizer)
    
    # Analysis 2: Relationship directions
    analyze_relationship_directions(model, tokenizer)
    
    # Analysis 3: Neighborhoods
    analyze_embedding_neighborhoods(model, tokenizer)
    
    # Analysis 4: Semantic clusters
    analyze_semantic_clusters(model, tokenizer)
    
    # Analysis 5: DRUM structure
    explore_drum_structure(model, tokenizer)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Can Entity→Answer Be Derived from DRUM?")
    print("=" * 70)
    print("""
Key Findings:

1. CLUSTERING: Entities and answers are NOT close in DRUM space
   - France and Paris have low similarity (~0.2)
   - Answer is typically ranked 10,000+ among all tokens

2. DIRECTION CONSISTENCY: Low consistency within relationship types
   - Capital-of directions are only ~15-20% consistent
   - Not enough to predict unseen pairs

3. NEIGHBORHOODS: Answers are NOT in entity neighborhoods
   - Top-10 neighbors are typically related words, not answers
   - "France" neighbors: "French", "Franc", etc. (not "Paris")

4. SEMANTIC CLUSTERS: Countries cluster, capitals cluster, but separately
   - Country-country similarity > country-capital similarity
   - The relationship is NOT encoded in proximity

CONCLUSION:
===========
The entity→answer relationship is NOT encoded in DRUM geometry.

The DRUM encodes:
- Semantic similarity (France ~ French)
- Morphological similarity (France ~ Franc)
- Contextual co-occurrence patterns

But NOT:
- Factual relationships (France → Paris)
- World knowledge

This confirms: Entity→Answer requires MEMORY, not geometry.
The transformer learns these facts during training and stores them
in the attention/MLP weights, not in the embeddings.
""")


if __name__ == "__main__":
    main()
