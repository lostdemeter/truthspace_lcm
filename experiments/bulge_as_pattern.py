#!/usr/bin/env python3
"""
Bulge Basis = Pattern Dimensions?
=================================

Hypothesis from Doc 119-120:
- Patterns ARE concepts (formal, casual, verbose, terse)
- They exist in the same φ-space as content

Today's Discovery:
- Trajectories = Geodesic + Bulge
- Bulge has wavelet-like basis functions
- 10 basis functions capture 87.5% variance

The Connection:
- Are the bulge basis functions the PATTERN dimensions?
- Does basis 0 = "continuation pattern"?
- Does basis 1 = "elaboration pattern"?

If true:
- Content = geodesic endpoints (Paris, Berlin)
- Pattern = bulge coefficients (formal, verbose)
- The bulge IS the "how to say it" from Doc 119!

Author: TruthSpace LCM Team
Date: 2026-01-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

PHI = 1.6180339887498949


def collect_diverse_trajectories(model, tokenizer, n_tokens: int = 8):
    """
    Collect trajectories with DIFFERENT patterns but SAME content type.
    
    If bulge = pattern, then:
    - Same content type → similar geodesic
    - Different pattern → different bulge
    """
    
    # Same content type (capitals), different expected patterns
    prompts_by_pattern = {
        "factual": [
            "The capital of France is",
            "The capital of Germany is",
        ],
        "question": [
            "What is the capital of France?",
            "What is the capital of Germany?",
        ],
        "elaborate": [
            "Tell me about the capital of France in detail.",
            "Tell me about the capital of Germany in detail.",
        ],
    }
    
    trajectories = []
    all_tokens = []
    patterns = []
    entities = []
    
    for pattern_type, pattern_prompts in prompts_by_pattern.items():
        for prompt in pattern_prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            
            hidden_states = []
            tokens = []
            
            for i in range(n_tokens):
                with torch.no_grad():
                    outputs = model(input_ids, output_hidden_states=True)
                    h = outputs.hidden_states[-1][0, -1, :]
                    hidden_states.append(h)
                    
                    next_token = outputs.logits[0, -1, :].argmax()
                    tokens.append(next_token.item())
                    
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            trajectories.append(torch.stack(hidden_states))
            all_tokens.append(tokens)
            patterns.append(pattern_type)
            
            # Extract entity
            if "France" in prompt:
                entities.append("France")
            elif "Germany" in prompt:
                entities.append("Germany")
            else:
                entities.append("unknown")
    
    return trajectories, all_tokens, patterns, entities


def compute_bulges_and_analyze(trajectories, patterns, entities, P):
    """
    Compute bulges and analyze if they correlate with patterns.
    """
    print("\n" + "=" * 70)
    print("Bulge vs Pattern Analysis")
    print("=" * 70)
    
    # Compute bulges
    all_bulges = []
    
    for traj in trajectories:
        traj_proj = traj @ P.T
        h_start = traj_proj[0]
        h_end = traj_proj[-1]
        
        n_steps = len(traj)
        bulges = []
        
        for j in range(n_steps):
            t = j / (n_steps - 1) if n_steps > 1 else 0
            h_geo = (1 - t) * h_start + t * h_end
            bulge = traj_proj[j] - h_geo
            bulges.append(bulge)
        
        all_bulges.append(torch.stack(bulges))
    
    # Get middle bulge as representative
    mid_bulges = []
    for b in all_bulges:
        mid_idx = len(b) // 2
        mid_bulges.append(b[mid_idx])
    
    mid_bulges = torch.stack(mid_bulges)
    
    # Analyze: Do same patterns have similar bulges?
    print("\n--- Bulge Similarity by Pattern ---")
    
    pattern_types = list(set(patterns))
    
    for p1 in pattern_types:
        for p2 in pattern_types:
            indices1 = [i for i, p in enumerate(patterns) if p == p1]
            indices2 = [i for i, p in enumerate(patterns) if p == p2]
            
            sims = []
            for i in indices1:
                for j in indices2:
                    if i != j:
                        b1 = mid_bulges[i]
                        b2 = mid_bulges[j]
                        sim = F.cosine_similarity(b1.unsqueeze(0), b2.unsqueeze(0)).item()
                        sims.append(sim)
            
            if sims:
                print(f"  {p1} vs {p2}: similarity = {np.mean(sims):.4f}")
    
    # Analyze: Do same entities have similar bulges (regardless of pattern)?
    print("\n--- Bulge Similarity by Entity ---")
    
    entity_types = list(set(entities))
    
    for e1 in entity_types:
        for e2 in entity_types:
            indices1 = [i for i, e in enumerate(entities) if e == e1]
            indices2 = [i for i, e in enumerate(entities) if e == e2]
            
            sims = []
            for i in indices1:
                for j in indices2:
                    if i != j:
                        b1 = mid_bulges[i]
                        b2 = mid_bulges[j]
                        sim = F.cosine_similarity(b1.unsqueeze(0), b2.unsqueeze(0)).item()
                        sims.append(sim)
            
            if sims:
                print(f"  {e1} vs {e2}: similarity = {np.mean(sims):.4f}")
    
    return all_bulges, mid_bulges


def decompose_bulge_into_pattern_content(bulges, patterns, entities, P):
    """
    Try to decompose bulge into pattern component and content component.
    
    If bulge = pattern × content:
    - Pattern component should be similar for same pattern type
    - Content component should be similar for same entity
    """
    print("\n" + "=" * 70)
    print("Bulge Decomposition: Pattern vs Content")
    print("=" * 70)
    
    # Stack all middle bulges
    mid_bulges = []
    for b in bulges:
        mid_idx = len(b) // 2
        mid_bulges.append(b[mid_idx])
    
    mid_bulges = torch.stack(mid_bulges)
    
    # SVD to find principal components
    U, S, Vt = torch.linalg.svd(mid_bulges, full_matrices=False)
    
    print(f"\nTop 5 singular values: {S[:5].tolist()}")
    
    # Project onto top components
    n_components = 5
    projections = mid_bulges @ Vt[:n_components, :].T
    
    print(f"\nProjections onto top {n_components} components:")
    
    for i, (proj, pattern, entity) in enumerate(zip(projections, patterns, entities)):
        print(f"  {pattern:10} {entity:10}: {proj[:3].tolist()}")
    
    # Check if any component correlates with pattern
    print("\n--- Component Correlation with Pattern ---")
    
    pattern_to_idx = {p: i for i, p in enumerate(set(patterns))}
    pattern_labels = torch.tensor([pattern_to_idx[p] for p in patterns], dtype=torch.float)
    
    for comp in range(n_components):
        comp_values = projections[:, comp]
        corr = torch.corrcoef(torch.stack([comp_values, pattern_labels]))[0, 1].item()
        print(f"  Component {comp}: correlation with pattern = {corr:.4f}")
    
    # Check if any component correlates with entity
    print("\n--- Component Correlation with Entity ---")
    
    entity_to_idx = {e: i for i, e in enumerate(set(entities))}
    entity_labels = torch.tensor([entity_to_idx[e] for e in entities], dtype=torch.float)
    
    for comp in range(n_components):
        comp_values = projections[:, comp]
        corr = torch.corrcoef(torch.stack([comp_values, entity_labels]))[0, 1].item()
        print(f"  Component {comp}: correlation with entity = {corr:.4f}")
    
    return Vt, projections


def test_pattern_transfer(model, tokenizer, trajectories, all_tokens, patterns, entities, P):
    """
    Test: Can we transfer a pattern from one entity to another?
    
    If bulge = pattern:
    - Take France's "factual" bulge
    - Apply to Germany's geodesic
    - Should get Germany's factual response
    """
    print("\n" + "=" * 70)
    print("Pattern Transfer Test")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data
    
    # Find France factual and Germany factual
    france_factual_idx = None
    germany_factual_idx = None
    
    for i, (pattern, entity) in enumerate(zip(patterns, entities)):
        if pattern == "factual" and entity == "France":
            france_factual_idx = i
        if pattern == "factual" and entity == "Germany":
            germany_factual_idx = i
    
    if france_factual_idx is None or germany_factual_idx is None:
        print("Could not find required trajectories")
        return
    
    # Get France's bulge
    france_traj = trajectories[france_factual_idx] @ P.T
    france_start = france_traj[0]
    france_end = france_traj[-1]
    
    france_bulges = []
    for j in range(len(france_traj)):
        t = j / (len(france_traj) - 1)
        h_geo = (1 - t) * france_start + t * france_end
        bulge = france_traj[j] - h_geo
        france_bulges.append(bulge)
    
    # Get Germany's geodesic
    germany_traj = trajectories[germany_factual_idx] @ P.T
    germany_start = germany_traj[0]
    germany_end = germany_traj[-1]
    
    # Apply France's bulge to Germany's geodesic
    print("\n--- Applying France's Pattern to Germany's Geodesic ---")
    
    transferred_tokens = []
    
    for j in range(len(germany_traj)):
        t = j / (len(germany_traj) - 1)
        
        # Germany's geodesic
        h_geo = (1 - t) * germany_start + t * germany_end
        
        # France's bulge
        france_bulge = france_bulges[j]
        
        # Transfer: Germany geodesic + France bulge
        h_transferred = h_geo + france_bulge
        
        # Decode
        h_full = h_transferred @ P
        logits = h_full @ lm_head.T
        token_id = logits.argmax().item()
        transferred_tokens.append(token_id)
    
    # Compare
    france_text = [tokenizer.decode([t]) for t in all_tokens[france_factual_idx]]
    germany_text = [tokenizer.decode([t]) for t in all_tokens[germany_factual_idx]]
    transferred_text = [tokenizer.decode([t]) for t in transferred_tokens]
    
    print(f"\nFrance actual:     {france_text}")
    print(f"Germany actual:    {germany_text}")
    print(f"Germany+Fr bulge:  {transferred_text}")
    
    # Check: Does transferred match Germany's actual?
    match_count = sum(1 for t, a in zip(transferred_tokens, all_tokens[germany_factual_idx]) if t == a)
    print(f"\nMatch with Germany actual: {match_count}/{len(transferred_tokens)}")


def synthesize_pattern_findings():
    """Synthesize findings about bulge as pattern."""
    print("\n" + "=" * 70)
    print("SYNTHESIS: Bulge as Pattern Dimension")
    print("=" * 70)
    print("""
Hypothesis from Doc 119-120:
  - Patterns ARE concepts in the same φ-space
  - Content = WHAT to say
  - Pattern = HOW to say it

Today's Discovery:
  - Trajectory = Geodesic + Bulge
  - Geodesic = content (endpoints)
  - Bulge = ???

The Connection:
  - If bulge correlates with PATTERN type → bulge IS the pattern
  - If bulge correlates with ENTITY → bulge IS the content
  - If bulge correlates with BOTH → bulge encodes both

Implications:

IF BULGE = PATTERN:
  - The wavelet basis functions ARE the pattern dimensions
  - Basis 0 = "continuation" (. It is the...)
  - Basis 1 = "elaboration" (detailed description)
  - Basis 2 = "enumeration" (list of facts)
  
  - Coefficients encode WHICH pattern, not WHAT content
  - Content is in the geodesic endpoints
  - Pattern is in the bulge direction

IF BULGE = CONTENT:
  - The wavelet basis functions encode entity-specific info
  - Paris has different bulge than Berlin
  - Pattern is in the geodesic shape

IF BULGE = BOTH:
  - Some components encode pattern
  - Some components encode content
  - Need to disentangle

This connects our discoveries:
  Doc 119-120: Patterns are dimensions
  Doc 177: Scaffolding vs Content
  Today: Geodesic + Bulge

The unified picture:
  GEODESIC = scaffold structure (predictable)
  BULGE = content + pattern (entity-specific + style-specific)
""")


def main():
    print("=" * 70)
    print("Bulge as Pattern Dimension Analysis")
    print("=" * 70)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Collect diverse trajectories
    print("\n--- Collecting Diverse Trajectories ---")
    trajectories, all_tokens, patterns, entities = collect_diverse_trajectories(
        model, tokenizer, n_tokens=6
    )
    
    print(f"Collected {len(trajectories)} trajectories")
    print(f"Patterns: {patterns}")
    print(f"Entities: {entities}")
    
    # Print what was generated
    print("\n--- Generated Responses ---")
    for i, (toks, pattern, entity) in enumerate(zip(all_tokens, patterns, entities)):
        text = [tokenizer.decode([t]) for t in toks]
        print(f"  {pattern:10} {entity:10}: {text}")
    
    # Compute projection matrix
    all_points = torch.cat(trajectories, dim=0)
    U, S, Vt = torch.linalg.svd(all_points, full_matrices=False)
    P = Vt[:100, :]
    
    # Analyze bulges
    all_bulges, mid_bulges = compute_bulges_and_analyze(trajectories, patterns, entities, P)
    
    # Decompose bulge
    Vt_bulge, projections = decompose_bulge_into_pattern_content(all_bulges, patterns, entities, P)
    
    # Test pattern transfer
    test_pattern_transfer(model, tokenizer, trajectories, all_tokens, patterns, entities, P)
    
    # Synthesis
    synthesize_pattern_findings()


if __name__ == "__main__":
    main()
