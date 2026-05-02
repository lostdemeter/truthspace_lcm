#!/usr/bin/env python3
"""
Find the Capital Axis: Geometric Relationship Discovery
=========================================================

From Doc 160: "The neural network discovered the optimal structure through optimization."

From our experiments:
- France→Paris angle: 77.8°
- Germany→Berlin angle: 75.6°
- The angle is consistent (~77°), but the direction varies

Hypothesis: There IS a "capital axis" - a direction in embedding space that,
when combined with a country embedding, points toward its capital.

The challenge: The transformation isn't a single vector, but it might be
decomposable into:
  1. A universal "capital-of" component
  2. A country-specific component

Let's try to find this decomposition.

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
    """Get embedding for a word."""
    ids = tokenizer.encode(word, add_special_tokens=False)
    if ids:
        return embeddings[ids[0]]
    return None


def compute_rotation_axis(v1: torch.Tensor, v2: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Compute the rotation axis and angle from v1 to v2.
    
    The rotation axis is perpendicular to both v1 and v2.
    In high dimensions, we use the rejection of v2 from v1.
    """
    # Normalize
    v1_norm = v1 / v1.norm()
    v2_norm = v2 / v2.norm()
    
    # Angle
    cos_angle = (v1_norm @ v2_norm).item()
    angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
    
    # Axis: perpendicular component of v2 relative to v1
    # v2 = (v2 · v1_norm) * v1_norm + perpendicular
    parallel = (v2_norm @ v1_norm) * v1_norm
    perpendicular = v2_norm - parallel
    
    if perpendicular.norm() > 1e-6:
        axis = perpendicular / perpendicular.norm()
    else:
        axis = torch.zeros_like(v1)
    
    return axis, angle


def find_capital_axis_svd(embeddings, tokenizer):
    """
    Use SVD to find the principal "capital-of" direction.
    
    Stack all (capital - country) vectors and find the principal component.
    """
    print("=" * 70)
    print("FINDING CAPITAL AXIS VIA SVD")
    print("=" * 70)
    
    pairs = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
        ("Japan", "Tokyo"),
        ("China", "Beijing"),
        ("Poland", "Warsaw"),
        ("Egypt", "Cairo"),
        ("Greece", "Athens"),
        ("Sweden", "Stockholm"),
        ("Norway", "Oslo"),
        ("Austria", "Vienna"),
        ("Portugal", "Lisbon"),
        ("Brazil", "Brasilia"),
        ("India", "Delhi"),
        ("Russia", "Moscow"),
        ("Canada", "Ottawa"),
        ("Australia", "Canberra"),
        ("Mexico", "Mexico"),
        ("Argentina", "Aires"),
    ]
    
    # Collect transformation vectors
    transforms = []
    valid_pairs = []
    
    for country, capital in pairs:
        country_emb = get_embedding(embeddings, tokenizer, country)
        capital_emb = get_embedding(embeddings, tokenizer, " " + capital)  # With space
        
        if country_emb is not None and capital_emb is not None:
            transform = capital_emb - country_emb
            transforms.append(transform)
            valid_pairs.append((country, capital))
    
    if len(transforms) < 2:
        print("Not enough data")
        return None
    
    # Stack and SVD
    transform_matrix = torch.stack(transforms)
    print(f"\nTransform matrix shape: {transform_matrix.shape}")
    
    U, S, Vt = torch.linalg.svd(transform_matrix, full_matrices=False)
    
    # Principal direction
    principal_axis = Vt[0]
    
    # How much variance does the first component explain?
    total_var = (S ** 2).sum().item()
    first_var = (S[0] ** 2).item()
    var_explained = first_var / total_var
    
    print(f"\nSingular values (top 10): {S[:10].tolist()}")
    print(f"Variance explained by first component: {var_explained * 100:.1f}%")
    
    # Test: project each transform onto principal axis
    print(f"\nProjection onto principal axis:")
    for i, (country, capital) in enumerate(valid_pairs[:10]):
        proj = (transforms[i] @ principal_axis).item()
        print(f"  {country}→{capital}: {proj:.4f}")
    
    return principal_axis, S, Vt


def test_principal_axis_prediction(embeddings, tokenizer, principal_axis):
    """
    Test: Can we predict capitals using the principal axis?
    
    Method: country_emb + scale * principal_axis → nearest token
    """
    print("\n" + "=" * 70)
    print("TESTING PRINCIPAL AXIS PREDICTION")
    print("=" * 70)
    
    test_pairs = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Japan", "Tokyo"),
        ("United", "Washington"),  # United States → Washington
    ]
    
    # Find optimal scale
    scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for scale in scales:
        correct = 0
        total = 0
        
        print(f"\n--- Scale = {scale} ---")
        
        for country, expected in test_pairs:
            country_emb = get_embedding(embeddings, tokenizer, country)
            if country_emb is None:
                continue
            
            # Predict
            predicted_emb = country_emb + scale * principal_axis
            
            # Find nearest
            distances = (embeddings - predicted_emb.unsqueeze(0)).norm(dim=1)
            nearest_idx = distances.argmin().item()
            nearest_token = tokenizer.decode([nearest_idx]).strip()
            
            top5_indices = distances.argsort()[:5]
            top5_tokens = [tokenizer.decode([idx.item()]).strip() for idx in top5_indices]
            
            is_correct = expected.lower() in nearest_token.lower()
            if is_correct:
                correct += 1
            total += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  {country} + axis → {nearest_token} (expected: {expected}) {status}")
            print(f"    Top 5: {top5_tokens}")
        
        print(f"  Accuracy: {correct}/{total}")


def find_rotation_based_axis(embeddings, tokenizer):
    """
    Alternative: Find the axis by analyzing rotation structure.
    
    From Doc 180: The rotation angle is ~77° for capital-of.
    The axis should be orthogonal to both country and capital embeddings.
    """
    print("\n" + "=" * 70)
    print("FINDING AXIS VIA ROTATION ANALYSIS")
    print("=" * 70)
    
    pairs = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Japan", "Tokyo"),
    ]
    
    axes = []
    angles = []
    
    for country, capital in pairs:
        country_emb = get_embedding(embeddings, tokenizer, country)
        capital_emb = get_embedding(embeddings, tokenizer, " " + capital)
        
        if country_emb is not None and capital_emb is not None:
            axis, angle = compute_rotation_axis(country_emb, capital_emb)
            axes.append(axis)
            angles.append(angle)
            
            print(f"\n{country}→{capital}:")
            print(f"  Angle: {angle:.1f}°")
    
    print(f"\nMean angle: {np.mean(angles):.1f}°")
    
    # Are the axes similar?
    print(f"\nAxis similarities:")
    for i in range(len(axes)):
        for j in range(len(axes)):
            if i < j:
                sim = (axes[i] @ axes[j]).item()
                print(f"  {pairs[i][0]}-{pairs[j][0]}: {sim:.4f}")
    
    # Mean axis
    mean_axis = torch.stack(axes).mean(dim=0)
    mean_axis = mean_axis / mean_axis.norm()
    
    return mean_axis, np.mean(angles)


def test_rotation_prediction(embeddings, tokenizer, axis, angle):
    """
    Test: Can we predict capitals by rotating country embeddings?
    
    Method: Rotate country_emb by angle degrees toward axis
    """
    print("\n" + "=" * 70)
    print("TESTING ROTATION-BASED PREDICTION")
    print("=" * 70)
    
    test_pairs = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Japan", "Tokyo"),
    ]
    
    angle_rad = angle * np.pi / 180
    
    for country, expected in test_pairs:
        country_emb = get_embedding(embeddings, tokenizer, country)
        if country_emb is None:
            continue
        
        # Normalize
        country_norm = country_emb / country_emb.norm()
        
        # Rodrigues rotation formula (simplified for high-dim)
        # v_rot = v*cos(θ) + (axis × v)*sin(θ) + axis*(axis·v)*(1-cos(θ))
        # In high-dim, we approximate by moving along the axis direction
        
        # Simple approach: move toward axis by angle amount
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Component of country along axis
        parallel = (country_norm @ axis) * axis
        perpendicular = country_norm - parallel
        
        # Rotate in the plane spanned by country and axis
        rotated = cos_a * country_norm + sin_a * axis
        rotated = rotated / rotated.norm()
        
        # Scale back to original magnitude
        predicted_emb = rotated * country_emb.norm()
        
        # Find nearest
        distances = (embeddings - predicted_emb.unsqueeze(0)).norm(dim=1)
        nearest_idx = distances.argmin().item()
        nearest_token = tokenizer.decode([nearest_idx]).strip()
        
        top5_indices = distances.argsort()[:5]
        top5_tokens = [tokenizer.decode([idx.item()]).strip() for idx in top5_indices]
        
        is_correct = expected.lower() in nearest_token.lower()
        status = "✓" if is_correct else "✗"
        
        print(f"\n{country} rotated by {angle:.1f}° → {nearest_token} (expected: {expected}) {status}")
        print(f"  Top 5: {top5_tokens}")


def analyze_hidden_state_geometry(model, tokenizer, device):
    """
    The key insight: The hidden state after "The capital of France is"
    correctly predicts Paris.
    
    Can we understand the geometry of this hidden state?
    """
    print("\n" + "=" * 70)
    print("HIDDEN STATE GEOMETRY ANALYSIS")
    print("=" * 70)
    
    lm_head = model.lm_head.weight.data.float().cpu()
    
    # Get hidden states for different prompts
    prompts = [
        ("The capital of France is", "Paris"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of Italy is", "Rome"),
    ]
    
    hidden_states = []
    
    for prompt, expected in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :].float().cpu()
        
        hidden_states.append({
            'prompt': prompt,
            'expected': expected,
            'hidden': hidden,
        })
    
    # Compute pairwise similarities between hidden states
    print("\nHidden state similarities:")
    for i, h1 in enumerate(hidden_states):
        for j, h2 in enumerate(hidden_states):
            if i < j:
                sim = torch.nn.functional.cosine_similarity(
                    h1['hidden'].unsqueeze(0),
                    h2['hidden'].unsqueeze(0)
                ).item()
                print(f"  {h1['expected']}-{h2['expected']}: {sim:.4f}")
    
    # What's the common component?
    mean_hidden = torch.stack([h['hidden'] for h in hidden_states]).mean(dim=0)
    
    print(f"\nMean hidden state magnitude: {mean_hidden.norm().item():.2f}")
    
    # What does the mean hidden state predict?
    logits = mean_hidden @ lm_head.T
    top_idx = logits.argmax().item()
    top_token = tokenizer.decode([top_idx])
    
    print(f"Mean hidden state top prediction: {repr(top_token)}")
    
    # Difference from mean
    print(f"\nDifference from mean:")
    for h in hidden_states:
        diff = h['hidden'] - mean_hidden
        diff_logits = diff @ lm_head.T
        top_diff_idx = diff_logits.argmax().item()
        top_diff_token = tokenizer.decode([top_diff_idx])
        
        print(f"  {h['expected']}: diff points toward {repr(top_diff_token)}")


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
    
    # Method 1: SVD on transformation vectors
    principal_axis, S, Vt = find_capital_axis_svd(embeddings, tokenizer)
    
    # Test principal axis
    if principal_axis is not None:
        test_principal_axis_prediction(embeddings, tokenizer, principal_axis)
    
    # Method 2: Rotation analysis
    rotation_axis, mean_angle = find_rotation_based_axis(embeddings, tokenizer)
    
    # Test rotation
    test_rotation_prediction(embeddings, tokenizer, rotation_axis, mean_angle)
    
    # Analyze hidden state geometry
    analyze_hidden_state_geometry(model, tokenizer, device)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
