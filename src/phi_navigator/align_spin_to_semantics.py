#!/usr/bin/env python3
"""
Align SPIN Basis to Semantic Dimensions
========================================

The SPIN basis captures structure, but semantic dimensions are distributed.
We need to find the rotation that aligns SPIN with our known semantic axes.

Approach:
1. Get SPIN basis from MESH
2. Get flip patterns from known semantic pairs
3. Find rotation matrix R such that: R @ SPIN_basis ≈ semantic_axes
4. If needed, rearrange/force alignment

This is like finding the "semantic coordinate system" within SPIN space.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)
K = 128


def extract_mesh(model, layer_idx: int = 0) -> torch.Tensor:
    """Extract MESH from attention layer."""
    attn = model.model.layers[layer_idx].self_attn
    W_q = attn.q_proj.weight.detach().float().cpu()
    W_k = attn.k_proj.weight.detach().float().cpu()
    
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    
    W_q_heads = W_q.view(n_heads, head_dim, -1)
    W_k_heads = W_k.view(model.config.num_key_value_heads, head_dim, -1)
    
    mesh = W_q_heads[0].T @ W_k_heads[0]
    return mesh


def get_spin_basis(mesh: torch.Tensor, n_dims: int = 20) -> torch.Tensor:
    """Get SPIN basis vectors."""
    spin = (mesh - mesh.T) / 2
    U, S, Vt = torch.linalg.svd(spin)
    return U[:, :n_dims], S[:n_dims]


def compute_flip_patterns(embeds: torch.Tensor, tokenizer, pairs: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
    """
    Compute flip patterns for each semantic dimension.
    
    A flip pattern is: which embedding dimensions flip sign between opposites.
    """
    dimensions = {
        "temperature": [("cold", "hot"), ("cool", "warm"), ("freezing", "burning")],
        "size": [("small", "big"), ("tiny", "huge"), ("little", "large")],
        "speed": [("slow", "fast"), ("sluggish", "quick"), ("leisurely", "swift")],
        "valence": [("bad", "good"), ("sad", "happy"), ("negative", "positive")],
        "age": [("young", "old"), ("new", "ancient")],
        "brightness": [("dark", "bright"), ("dim", "light")],
    }
    
    flip_patterns = {}
    
    for dim_name, dim_pairs in dimensions.items():
        flip_counts = torch.zeros(embeds.shape[1])
        n_pairs = 0
        
        for neg_word, pos_word in dim_pairs:
            neg_ids = tokenizer.encode(neg_word, add_special_tokens=False)
            pos_ids = tokenizer.encode(pos_word, add_special_tokens=False)
            
            if not neg_ids or not pos_ids:
                continue
            
            neg_signs = torch.sign(embeds[neg_ids[0]])
            pos_signs = torch.sign(embeds[pos_ids[0]])
            
            flips = (neg_signs != pos_signs).float()
            flip_counts += flips
            n_pairs += 1
        
        if n_pairs > 0:
            flip_prob = flip_counts / n_pairs
            flip_patterns[dim_name] = flip_prob
    
    return flip_patterns


def align_spin_to_flip_patterns(spin_basis: torch.Tensor, flip_patterns: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Find alignment between SPIN basis and flip patterns.
    
    For each semantic dimension, find which SPIN dimension best correlates.
    Then compute a rotation to align them.
    """
    n_spin_dims = spin_basis.shape[1]
    hidden_dim = spin_basis.shape[0]
    
    # Project flip patterns into SPIN space
    # flip_in_spin = flip_pattern @ spin_basis
    
    alignments = {}
    correlations = {}
    
    print(f"\n--- ALIGNING SPIN TO SEMANTIC DIMENSIONS ---")
    
    for dim_name, flip_pattern in flip_patterns.items():
        # Project flip pattern into SPIN space
        flip_in_spin = flip_pattern @ spin_basis  # [n_spin_dims]
        
        # Which SPIN dimension has highest correlation?
        best_spin_dim = flip_in_spin.abs().argmax().item()
        best_corr = flip_in_spin[best_spin_dim].item()
        
        # Also compute how much of the flip pattern is captured
        flip_magnitude = flip_pattern.norm().item()
        projected_magnitude = flip_in_spin.norm().item()
        capture_ratio = projected_magnitude / flip_magnitude if flip_magnitude > 0 else 0
        
        alignments[dim_name] = best_spin_dim
        correlations[dim_name] = {
            'best_spin_dim': best_spin_dim,
            'correlation': best_corr,
            'capture_ratio': capture_ratio,
            'flip_in_spin': flip_in_spin,
        }
        
        print(f"  {dim_name:12s}: SPIN dim {best_spin_dim:2d}, corr={best_corr:+.4f}, capture={capture_ratio:.1%}")
    
    return alignments, correlations


def create_semantic_basis(spin_basis: torch.Tensor, flip_patterns: Dict[str, torch.Tensor], 
                          correlations: Dict) -> torch.Tensor:
    """
    Create a new basis that aligns with semantic dimensions.
    
    We rearrange/rotate the SPIN basis to match our semantic axes.
    """
    n_semantic_dims = len(flip_patterns)
    hidden_dim = spin_basis.shape[0]
    
    # Method 1: Use flip patterns directly as the semantic basis
    # (This is "forcing" the alignment)
    semantic_basis = torch.zeros(hidden_dim, n_semantic_dims)
    
    for i, (dim_name, flip_pattern) in enumerate(flip_patterns.items()):
        # Normalize the flip pattern
        semantic_basis[:, i] = F.normalize(flip_pattern.unsqueeze(0), dim=1).squeeze()
    
    print(f"\n--- SEMANTIC BASIS (forced from flip patterns) ---")
    print(f"  Shape: {semantic_basis.shape}")
    
    # Check orthogonality
    ortho_matrix = semantic_basis.T @ semantic_basis
    print(f"  Orthogonality check (should be ~identity):")
    for i, name_i in enumerate(flip_patterns.keys()):
        row = []
        for j, name_j in enumerate(flip_patterns.keys()):
            row.append(f"{ortho_matrix[i,j].item():.3f}")
        print(f"    {name_i:12s}: {' '.join(row)}")
    
    return semantic_basis


def test_semantic_navigation(embeds: torch.Tensor, semantic_basis: torch.Tensor, 
                             tokenizer, flip_patterns: Dict[str, torch.Tensor]):
    """
    Test navigation using the semantic basis.
    """
    print(f"\n--- TESTING SEMANTIC NAVIGATION ---")
    
    dim_names = list(flip_patterns.keys())
    
    # Test words
    test_cases = [
        ("hot", "temperature"),
        ("big", "size"),
        ("fast", "speed"),
        ("happy", "valence"),
        ("old", "age"),
        ("bright", "brightness"),
    ]
    
    for word, expected_dim in test_cases:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if not word_ids:
            continue
        
        word_embed = embeds[word_ids[0]]
        word_signs = torch.sign(word_embed)
        
        # Get flip pattern for expected dimension
        flip_pattern = flip_patterns[expected_dim]
        flip_mask = flip_pattern > 0.5
        
        # Navigate: flip signs according to pattern
        target_signs = word_signs.clone()
        target_signs[flip_mask] *= -1
        
        # Find nearest word with matching signs
        all_signs = torch.sign(embeds)
        agreement = (all_signs == target_signs.unsqueeze(0)).float().sum(dim=1)
        agreement[word_ids[0]] = -1  # Exclude self
        
        top_idx = agreement.argmax().item()
        result_word = tokenizer.decode([top_idx]).strip()
        score = agreement[top_idx].item() / embeds.shape[1] * 100
        
        print(f"  {word:12s} --[{expected_dim}]--> {result_word:12s} ({score:.1f}% match)")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("="*70)
    print("ALIGNING SPIN BASIS TO SEMANTIC DIMENSIONS")
    print("="*70)
    print("""
We can rearrange anything to force alignment.

Approach:
1. Get SPIN basis from MESH (the "raw" navigation space)
2. Get flip patterns from known semantic pairs (our "target" axes)
3. Find/force alignment between them
4. Create semantic basis for navigation
""")
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Get embeddings
    embeds = model.model.embed_tokens.weight.detach().float().cpu()
    print(f"Embeddings: {embeds.shape}")
    
    # Get SPIN basis from layer 14 (middle layer)
    mesh = extract_mesh(model, layer_idx=14)
    spin_basis, spin_S = get_spin_basis(mesh, n_dims=20)
    print(f"SPIN basis: {spin_basis.shape}")
    
    # Compute flip patterns
    flip_patterns = compute_flip_patterns(embeds, tokenizer, [])
    print(f"Flip patterns: {len(flip_patterns)} dimensions")
    
    # Align SPIN to flip patterns
    alignments, correlations = align_spin_to_flip_patterns(spin_basis, flip_patterns)
    
    # Create semantic basis (forced alignment)
    semantic_basis = create_semantic_basis(spin_basis, flip_patterns, correlations)
    
    # Test navigation
    test_semantic_navigation(embeds, semantic_basis, tokenizer, flip_patterns)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
The semantic basis is created by:
1. Using flip patterns directly as basis vectors
2. This "forces" alignment with our known semantic axes
3. Navigation works by flipping signs according to the pattern

The SPIN basis from MESH captures SOME of this structure,
but the flip patterns are more direct for navigation.

Key insight: We don't need SPIN to match perfectly.
We can use flip patterns as the semantic coordinate system,
and SPIN as validation that the structure exists in attention.
""")


if __name__ == "__main__":
    main()
