#!/usr/bin/env python3
"""
Zeta-Anchored MESH Extraction
==============================

From Doc 143: Use zeta function as anchor to compare extracted structures.

The approach:
1. Extract attention MESH (W_q.T @ W_k) from each layer
2. Encode MESH into (sign, level) pairs
3. Anchor using critical line (level 0) as reference
4. Compare to embedding sign flip patterns

The zeta anchor:
- Level 0 = φ^0 = 1 (critical line, balance point)
- Errors cancel symmetrically around critical line
- ζ(s) = ζ(1-s) symmetry maps to φ and 1/φ symmetry
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)
K = 128  # Level quantization factor


def extract_mesh(model, layer_idx: int = 0) -> torch.Tensor:
    """
    Extract MESH = W_q.T @ W_k from attention layer.
    
    This is the "comb" that defines how Q and K interact.
    """
    # Qwen2 attention structure
    attn = model.model.layers[layer_idx].self_attn
    
    # Get Q and K projections
    W_q = attn.q_proj.weight.detach().float()  # [hidden, hidden]
    W_k = attn.k_proj.weight.detach().float()  # [kv_hidden, hidden]
    
    # MESH = W_q.T @ W_k
    # But dimensions may not match for GQA, so we need to handle that
    # For Qwen2-7B: Q is [3584, 3584], K is [512, 3584] (GQA with 8 KV heads)
    
    print(f"  W_q shape: {W_q.shape}")
    print(f"  W_k shape: {W_k.shape}")
    
    # For GQA, we need to expand K to match Q
    # Or we can compute the effective MESH differently
    # Let's compute the per-head MESH
    
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // n_heads
    
    print(f"  n_heads: {n_heads}, n_kv_heads: {n_kv_heads}, head_dim: {head_dim}")
    
    # Reshape to per-head
    W_q_heads = W_q.view(n_heads, head_dim, -1)  # [n_heads, head_dim, hidden]
    W_k_heads = W_k.view(n_kv_heads, head_dim, -1)  # [n_kv_heads, head_dim, hidden]
    
    # For GQA, each KV head serves multiple Q heads
    heads_per_kv = n_heads // n_kv_heads
    
    # Compute MESH for first head as example
    q_head = W_q_heads[0]  # [head_dim, hidden]
    k_head = W_k_heads[0]  # [head_dim, hidden]
    
    # MESH = Q.T @ K for this head pair
    mesh = q_head.T @ k_head  # [hidden, hidden]
    
    return mesh


def encode_to_phi_lattice(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode tensor to φ-lattice (sign, level) pairs.
    """
    # Move to CPU to avoid OOM
    tensor = tensor.cpu()
    
    signs = torch.sign(tensor)
    signs[signs == 0] = 1
    
    magnitudes = tensor.abs().clamp(min=1e-45)
    levels = torch.round(K * torch.log(magnitudes) / LOG_PHI)
    
    return signs.to(torch.int8), levels.to(torch.int16)


def analyze_mesh_structure(mesh: torch.Tensor):
    """
    Analyze the MESH structure using zeta-aligned principles.
    """
    print(f"\n--- MESH STRUCTURE ANALYSIS ---")
    print(f"  Shape: {mesh.shape}")
    print(f"  Mean: {mesh.mean().item():.6f}")
    print(f"  Std: {mesh.std().item():.6f}")
    print(f"  Min: {mesh.min().item():.6f}")
    print(f"  Max: {mesh.max().item():.6f}")
    
    # Encode to φ-lattice
    signs, levels = encode_to_phi_lattice(mesh)
    
    print(f"\n  φ-Lattice encoding:")
    print(f"    Level range: [{levels.min().item()}, {levels.max().item()}]")
    print(f"    Level mean: {levels.float().mean().item():.1f}")
    print(f"    Positive signs: {(signs > 0).sum().item()} / {signs.numel()}")
    
    # Critical line analysis (level 0 = φ^0 = 1)
    # How many values are near the critical line?
    near_critical = (levels.abs() < 10).sum().item()
    print(f"    Near critical line (|level| < 10): {near_critical} ({near_critical/levels.numel()*100:.1f}%)")
    
    # Symmetry analysis (zeta functional equation)
    # ζ(s) = ζ(1-s) means values should be symmetric around critical line
    pos_levels = levels[levels > 0].float()
    neg_levels = levels[levels < 0].float()
    
    if len(pos_levels) > 0 and len(neg_levels) > 0:
        pos_mean = pos_levels.mean().item()
        neg_mean = neg_levels.abs().mean().item()
        symmetry = 1 - abs(pos_mean - neg_mean) / max(pos_mean, neg_mean)
        print(f"    Level symmetry: {symmetry*100:.1f}% (pos_mean={pos_mean:.1f}, neg_mean={neg_mean:.1f})")
    
    # Decompose into MASS (symmetric) and SPIN (antisymmetric)
    mass = (mesh + mesh.T) / 2
    spin = (mesh - mesh.T) / 2
    
    mass_energy = (mass ** 2).sum().item()
    spin_energy = (spin ** 2).sum().item()
    total_energy = mass_energy + spin_energy
    
    print(f"\n  MASS/SPIN decomposition:")
    print(f"    MASS energy: {mass_energy/total_energy*100:.1f}%")
    print(f"    SPIN energy: {spin_energy/total_energy*100:.1f}%")
    
    # SVD of MESH
    U, S, Vt = torch.linalg.svd(mesh)
    
    print(f"\n  SVD analysis:")
    print(f"    Top 5 singular values: {S[:5].tolist()}")
    
    # Check for φ-scaling in singular values
    ratios = []
    for i in range(min(10, len(S) - 1)):
        if S[i+1] > 1e-6:
            ratio = S[i].item() / S[i+1].item()
            ratios.append(ratio)
            phi_diff = abs(ratio - PHI)
            marker = "≈φ" if phi_diff < 0.2 else ""
            print(f"    S_{i}/S_{i+1} = {ratio:.4f} {marker}")
    
    return signs, levels, mass, spin


def compare_mesh_to_embeddings(mesh_signs: torch.Tensor, mesh_levels: torch.Tensor,
                                embed_signs: torch.Tensor, embed_levels: torch.Tensor):
    """
    Compare MESH structure to embedding structure.
    
    The anchor is the critical line (level 0).
    """
    print(f"\n--- MESH vs EMBEDDING COMPARISON ---")
    
    # Both should have similar level distributions if they share the same φ-structure
    mesh_level_dist = mesh_levels.float().histc(bins=50, min=-2000, max=0)
    embed_level_dist = embed_levels.float().histc(bins=50, min=-2000, max=0)
    
    # Normalize
    mesh_level_dist = mesh_level_dist / mesh_level_dist.sum()
    embed_level_dist = embed_level_dist / embed_level_dist.sum()
    
    # KL divergence (how different are the distributions?)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    kl_div = (mesh_level_dist * torch.log((mesh_level_dist + eps) / (embed_level_dist + eps))).sum()
    
    print(f"  Level distribution KL divergence: {kl_div.item():.4f}")
    print(f"  (Lower = more similar)")
    
    # Sign distribution comparison
    mesh_pos_ratio = (mesh_signs > 0).float().mean().item()
    embed_pos_ratio = (embed_signs > 0).float().mean().item()
    
    print(f"  MESH positive sign ratio: {mesh_pos_ratio*100:.1f}%")
    print(f"  Embedding positive sign ratio: {embed_pos_ratio*100:.1f}%")
    
    # Critical line alignment
    mesh_critical = (mesh_levels.abs() < 50).float().mean().item()
    embed_critical = (embed_levels.abs() < 50).float().mean().item()
    
    print(f"  MESH near critical line: {mesh_critical*100:.1f}%")
    print(f"  Embedding near critical line: {embed_critical*100:.1f}%")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("="*70)
    print("ZETA-ANCHORED MESH EXTRACTION")
    print("="*70)
    print("""
From Doc 143: Use zeta function as anchor to compare structures.

1. Extract attention MESH (W_q.T @ W_k)
2. Encode to (sign, level) pairs
3. Anchor using critical line (level 0)
4. Compare to embedding sign patterns
""")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model.eval()
    
    # Extract embeddings
    print("\n--- EMBEDDING ANALYSIS ---")
    embeds = model.model.embed_tokens.weight.detach().float()
    embed_signs, embed_levels = encode_to_phi_lattice(embeds)
    
    print(f"  Embedding shape: {embeds.shape}")
    print(f"  Level range: [{embed_levels.min().item()}, {embed_levels.max().item()}]")
    print(f"  Level mean: {embed_levels.float().mean().item():.1f}")
    
    # Extract and analyze MESH from multiple layers
    for layer_idx in [0, 14, 27]:  # First, middle, last
        print(f"\n{'='*70}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*70}")
        
        mesh = extract_mesh(model, layer_idx)
        mesh_signs, mesh_levels, mass, spin = analyze_mesh_structure(mesh)
        
        # Compare to embeddings
        compare_mesh_to_embeddings(mesh_signs, mesh_levels, embed_signs, embed_levels)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
The zeta anchor (critical line at level 0) provides a reference point
for comparing the MESH structure to the embedding structure.

If they share the same φ-lattice geometry:
- Level distributions should be similar
- Sign ratios should be similar
- Both should cluster around the critical line

This would validate that attention and embeddings live in the same
geometric space, and sign navigation can work across both.
""")


if __name__ == "__main__":
    main()
