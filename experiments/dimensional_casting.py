#!/usr/bin/env python3
"""
Dimensional Casting: Downcast and Upcast for Entity→Answer
===========================================================

Inspired by dimensional_downcasting project:
- Downcasting: ∞D → 1D via moment projection (pure math)
- Upcasting: 1D → ∞D via radiance field (requires training)

Our problem:
- Entity embedding: 3584D
- Answer embedding: 3584D
- Relationship: Complex in 3584D, maybe simple in lower D?

Hypothesis:
1. DOWNCAST: Project entity and answer to a low-D space where
   the relationship becomes simple (e.g., linear, or even identity)
   
2. UPCAST: Given entity in low-D, predict answer in low-D,
   then upcast back to 3584D

The key insight from zeta zeros:
- N_smooth(t_n) ≈ n - 0.5 (a simple relationship in the right projection)
- The "right projection" makes the complex relationship trivial

For us:
- Maybe there's a projection where France→Paris becomes trivial
- The projection might be pattern-specific (capital-of vs opposite-of)

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


def explore_downcasting(model, tokenizer):
    """
    Explore dimensional downcasting for entity→answer mapping.
    
    The idea: Find a projection P such that:
      P @ entity_embed ≈ P @ answer_embed (or some simple transform)
    """
    print("\n" + "=" * 70)
    print("Dimensional Downcasting Exploration")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    lm_head = model.lm_head.weight.data
    
    # Capital pairs
    pairs = [
        ("France", " Paris"),
        ("Germany", " Berlin"),
        ("Italy", " Rome"),
        ("Spain", " Madrid"),
        ("Japan", " Tokyo"),
        ("China", " Beijing"),
        ("Poland", " Warsaw"),
        ("Brazil", " Brasilia"),
    ]
    
    # Get embeddings
    entities = []
    answers = []
    
    for entity, answer in pairs:
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        
        entities.append(embed[entity_ids[0]])
        answers.append(embed[answer_ids[0]])
    
    E = torch.stack(entities)  # [n, 3584]
    A = torch.stack(answers)   # [n, 3584]
    
    print(f"Entity embeddings shape: {E.shape}")
    print(f"Answer embeddings shape: {A.shape}")
    
    # Method 1: SVD of combined matrix
    # Find the principal directions of [E; A]
    print("\n--- Method 1: SVD of Combined Matrix ---")
    
    combined = torch.cat([E, A], dim=0)  # [2n, 3584]
    U, S, Vt = torch.linalg.svd(combined, full_matrices=False)
    
    print(f"Top 10 singular values: {S[:10].tolist()}")
    
    # Project onto top-k dimensions
    for k in [1, 2, 3, 5, 10]:
        P = Vt[:k, :]  # Projection matrix [k, 3584]
        
        E_proj = E @ P.T  # [n, k]
        A_proj = A @ P.T  # [n, k]
        
        # In this k-D space, what's the relationship?
        # Check if E_proj ≈ A_proj (identity)
        identity_error = (E_proj - A_proj).norm() / E_proj.norm()
        
        # Check if there's a linear transform: A_proj = E_proj @ W
        W = torch.linalg.lstsq(E_proj, A_proj).solution
        linear_error = (E_proj @ W - A_proj).norm() / A_proj.norm()
        
        print(f"  k={k}: identity_error={identity_error:.3f}, linear_error={linear_error:.3f}")
    
    # Method 2: Find projection that MINIMIZES entity-answer distance
    print("\n--- Method 2: Optimal Projection for Similarity ---")
    
    # We want: P @ entity ≈ P @ answer
    # This means: (entity - answer) should be orthogonal to P
    
    diff = E - A  # [n, 3584]
    
    # SVD of diff: find directions where entities and answers DIFFER
    U_diff, S_diff, Vt_diff = torch.linalg.svd(diff, full_matrices=False)
    
    print(f"Difference singular values: {S_diff[:10].tolist()}")
    
    # The LAST singular vectors are where E and A are SIMILAR
    # Project onto those!
    
    for k in [10, 50, 100, 500, 1000]:
        # Use the LAST k singular vectors (where E ≈ A)
        P_similar = Vt_diff[-k:, :]  # [k, 3584]
        
        E_proj = E @ P_similar.T
        A_proj = A @ P_similar.T
        
        similarity = F.cosine_similarity(E_proj, A_proj, dim=1).mean()
        
        print(f"  k={k} (similar dims): mean cosine similarity = {similarity:.4f}")
    
    return E, A, Vt, Vt_diff


def explore_upcasting(model, tokenizer, E, A, Vt_diff):
    """
    Explore dimensional upcasting: low-D → high-D.
    
    The idea: If we know the answer in low-D, can we upcast to 3584D?
    
    This is like: given the "capital-of" relationship in low-D,
    reconstruct the full answer embedding.
    """
    print("\n" + "=" * 70)
    print("Dimensional Upcasting Exploration")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    lm_head = model.lm_head.weight.data
    
    # Use the similar dimensions (where E ≈ A)
    k = 100
    P_similar = Vt_diff[-k:, :]  # [k, 3584]
    
    E_proj = E @ P_similar.T  # [n, k]
    A_proj = A @ P_similar.T  # [n, k]
    
    print(f"Projected to {k} dimensions")
    
    # In this space, E_proj ≈ A_proj
    # Can we use E_proj to find the answer?
    
    # Method 1: Nearest neighbor in projected space
    print("\n--- Method 1: Nearest Neighbor in Projected Space ---")
    
    # Project all embeddings
    all_proj = embed @ P_similar.T  # [vocab, k]
    
    correct = 0
    for i in range(len(E)):
        # Find nearest neighbor to E_proj[i]
        dists = (all_proj - E_proj[i]).norm(dim=1)
        
        # Exclude the entity itself
        entity_idx = (embed - E[i]).norm(dim=1).argmin()
        dists[entity_idx] = float('inf')
        
        pred_idx = dists.argmin()
        pred_text = tokenizer.decode([pred_idx])
        
        answer_idx = (embed - A[i]).norm(dim=1).argmin()
        answer_text = tokenizer.decode([answer_idx])
        
        marker = "✓" if pred_idx == answer_idx else "✗"
        if pred_idx == answer_idx:
            correct += 1
        
        print(f"  Entity {i}: pred={pred_text!r}, answer={answer_text!r} {marker}")
    
    print(f"  Accuracy: {correct}/{len(E)}")
    
    # Method 2: Learn upcast matrix
    print("\n--- Method 2: Learn Upcast Matrix ---")
    
    # Learn: A = E_proj @ U_up
    # Where U_up: [k, 3584]
    
    lambda_reg = 0.1
    n_proj = E_proj.shape[1]
    EtE = E_proj.T @ E_proj + lambda_reg * torch.eye(n_proj)
    EtA = E_proj.T @ A
    U_up = torch.linalg.solve(EtE, EtA)
    
    print(f"Upcast matrix shape: {U_up.shape}")
    
    # Test reconstruction
    A_recon = E_proj @ U_up
    
    recon_error = (A_recon - A).norm() / A.norm()
    print(f"Reconstruction error: {recon_error*100:.1f}%")
    
    # Can we predict the answer token?
    correct = 0
    for i in range(len(E)):
        a_pred = A_recon[i]
        
        # Find nearest token
        sims = F.cosine_similarity(a_pred.unsqueeze(0), embed)
        pred_idx = sims.argmax()
        pred_text = tokenizer.decode([pred_idx])
        
        answer_idx = (embed - A[i]).norm(dim=1).argmin()
        answer_text = tokenizer.decode([answer_idx])
        
        marker = "✓" if pred_idx == answer_idx else "✗"
        if pred_idx == answer_idx:
            correct += 1
        
        print(f"  Entity {i}: pred={pred_text!r}, answer={answer_text!r} {marker}")
    
    print(f"  Accuracy: {correct}/{len(E)}")


def explore_lm_head_projection(model, tokenizer):
    """
    Key insight: The lm_head IS a projection!
    
    lm_head: [vocab, 3584]
    logits = h @ lm_head.T
    
    This projects the hidden state onto vocab-dimensional space.
    The answer is the dimension with highest activation.
    
    What if we use lm_head as the downcasting projection?
    """
    print("\n" + "=" * 70)
    print("LM_HEAD as Downcasting Projection")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    lm_head = model.lm_head.weight.data
    
    pairs = [
        ("France", " Paris"),
        ("Germany", " Berlin"),
        ("Italy", " Rome"),
        ("Spain", " Madrid"),
    ]
    
    print("\n--- Entity and Answer in LM_HEAD Space ---")
    
    for entity, answer in pairs:
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        
        entity_embed = embed[entity_ids[0]]
        answer_embed = embed[answer_ids[0]]
        
        # Project onto lm_head (this gives "logits" for each vocab token)
        entity_logits = entity_embed @ lm_head.T  # [vocab]
        answer_logits = answer_embed @ lm_head.T  # [vocab]
        
        # What does the entity "predict"?
        entity_pred = entity_logits.argmax()
        entity_pred_text = tokenizer.decode([entity_pred])
        
        # What does the answer "predict"?
        answer_pred = answer_logits.argmax()
        answer_pred_text = tokenizer.decode([answer_pred])
        
        # How much does entity "activate" the answer?
        answer_activation = entity_logits[answer_ids[0]]
        answer_rank = (entity_logits > answer_activation).sum().item()
        
        print(f"  {entity}:")
        print(f"    Entity predicts: {entity_pred_text!r}")
        print(f"    Answer predicts: {answer_pred_text!r}")
        print(f"    Entity's activation of answer: rank {answer_rank}/{len(entity_logits)}")


def explore_quaternion_projection(model, tokenizer):
    """
    Inspired by Doc 044: Quaternion φ-Dial.
    
    What if the entity→answer relationship is a ROTATION in quaternion space?
    
    Quaternion: q = w + xi + yj + zk
    
    The 4D dial controls:
    - x: Style (what words)
    - y: Perspective (how framed)
    - z: Depth (how much detail)
    - w: Certainty (how sure)
    
    Maybe France→Paris is a rotation in this 4D space?
    """
    print("\n" + "=" * 70)
    print("Quaternion Projection Exploration")
    print("=" * 70)
    
    embed = model.model.embed_tokens.weight.data
    
    pairs = [
        ("France", " Paris"),
        ("Germany", " Berlin"),
        ("Italy", " Rome"),
        ("Spain", " Madrid"),
    ]
    
    # Get embeddings
    entities = []
    answers = []
    
    for entity, answer in pairs:
        entity_ids = tokenizer.encode(entity, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        
        entities.append(embed[entity_ids[0]])
        answers.append(embed[answer_ids[0]])
    
    E = torch.stack(entities)
    A = torch.stack(answers)
    
    # Project to 4D using PCA
    combined = torch.cat([E, A], dim=0)
    U, S, Vt = torch.linalg.svd(combined, full_matrices=False)
    
    P_4d = Vt[:4, :]  # [4, 3584]
    
    E_4d = E @ P_4d.T  # [n, 4]
    A_4d = A @ P_4d.T  # [n, 4]
    
    print("Entity and Answer in 4D space:")
    for i, (entity, answer) in enumerate(pairs):
        print(f"  {entity}: {E_4d[i].tolist()}")
        print(f"  {answer}: {A_4d[i].tolist()}")
        print()
    
    # Is there a consistent rotation?
    print("--- Rotation Analysis ---")
    
    # For quaternions, rotation is: q_answer = q_rotation * q_entity
    # In matrix form: A_4d = R @ E_4d.T
    
    # Find R using Procrustes
    # R = V @ U.T where E_4d.T @ A_4d = U @ S @ V.T
    
    M = E_4d.T @ A_4d
    U_r, S_r, Vt_r = torch.linalg.svd(M)
    R = Vt_r.T @ U_r.T
    
    print(f"Rotation matrix R:\n{R}")
    print(f"R is orthogonal: {torch.allclose(R @ R.T, torch.eye(4), atol=0.1)}")
    
    # Test rotation
    A_4d_pred = (R @ E_4d.T).T
    
    error = (A_4d_pred - A_4d).norm() / A_4d.norm()
    print(f"Rotation error: {error*100:.1f}%")
    
    # If error is low, we found a consistent rotation!


def synthesize_findings():
    """Synthesize what we learned about dimensional casting."""
    print("\n" + "=" * 70)
    print("SYNTHESIS: Dimensional Casting for Entity→Answer")
    print("=" * 70)
    print("""
Key Insights:

1. DOWNCASTING (∞D → low-D):
   - Project entity and answer to low-D space
   - In the "right" projection, the relationship becomes simple
   - The zeta zeros paper: N_smooth(t_n) ≈ n - 0.5 (trivial in right projection)

2. UPCASTING (low-D → ∞D):
   - Given the relationship in low-D, reconstruct high-D
   - This requires learning the upcast matrix
   - But the upcast matrix might be derivable from structure

3. LM_HEAD AS PROJECTION:
   - lm_head is already a projection: 3584D → vocab-D
   - The answer is the dimension with highest activation
   - This is the "natural" projection for next-token prediction

4. QUATERNION ROTATION:
   - In 4D, entity→answer might be a consistent rotation
   - This connects to Doc 044's quaternion φ-dial
   - Style, Perspective, Depth, Certainty

The Connection to Our Problem:
==============================
We found that W is low-rank (2-7 dimensions).
This means the entity→answer mapping IS a low-dimensional operation.

The question is: Can we find the "right" projection where:
  P @ entity ≈ P @ answer (or simple transform)

If yes, we can:
1. Downcast entity to low-D
2. Apply simple transform (identity, rotation, or linear)
3. Upcast back to 3584D
4. Find nearest token

This would give us PERFECT COVERAGE without training!
""")


def main():
    print("=" * 70)
    print("Dimensional Casting: Downcast and Upcast for Entity→Answer")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Exploration 1: Downcasting
    E, A, Vt, Vt_diff = explore_downcasting(model, tokenizer)
    
    # Exploration 2: Upcasting
    explore_upcasting(model, tokenizer, E, A, Vt_diff)
    
    # Exploration 3: LM_HEAD as projection
    explore_lm_head_projection(model, tokenizer)
    
    # Exploration 4: Quaternion projection
    explore_quaternion_projection(model, tokenizer)
    
    # Synthesis
    synthesize_findings()


if __name__ == "__main__":
    main()
