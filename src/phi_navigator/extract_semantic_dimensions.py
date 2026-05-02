#!/usr/bin/env python3
"""
Extract Semantic Dimensions from MESH
======================================

Semantic dimensions are tricky - they're not like other dimensions.

From the MESH analysis:
- MASS (symmetric): encodes similarity
- SPIN (antisymmetric): encodes navigation/transformation

The hypothesis:
- SPIN eigenvectors might BE the semantic dimensions
- They encode "which direction to flip" for transformations
- The eigenvalues encode "how strong" each dimension is

We'll compare extracted dimensions to our known semantic pairs
(hot/cold, big/small, etc.) to validate.
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
    """Extract MESH = W_q.T @ W_k from attention layer."""
    attn = model.model.layers[layer_idx].self_attn
    
    W_q = attn.q_proj.weight.detach().float().cpu()
    W_k = attn.k_proj.weight.detach().float().cpu()
    
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // n_heads
    
    W_q_heads = W_q.view(n_heads, head_dim, -1)
    W_k_heads = W_k.view(n_kv_heads, head_dim, -1)
    
    # First head pair
    q_head = W_q_heads[0]
    k_head = W_k_heads[0]
    
    mesh = q_head.T @ k_head
    return mesh


def decompose_mesh(mesh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose MESH into MASS (symmetric) and SPIN (antisymmetric).
    
    MASS = (M + M.T) / 2  -> similarity
    SPIN = (M - M.T) / 2  -> navigation
    """
    mass = (mesh + mesh.T) / 2
    spin = (mesh - mesh.T) / 2
    return mass, spin


def extract_spin_dimensions(spin: torch.Tensor, n_dims: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract semantic dimensions from SPIN matrix.
    
    SPIN is antisymmetric, so its eigenvalues are purely imaginary.
    We use SVD instead to get the principal directions.
    
    For antisymmetric matrices:
    - Singular values come in pairs
    - Each pair represents a 2D rotation plane
    - These planes might be semantic dimensions!
    """
    U, S, Vt = torch.linalg.svd(spin)
    
    # Take top n_dims
    top_U = U[:, :n_dims]
    top_S = S[:n_dims]
    top_Vt = Vt[:n_dims, :]
    
    return top_U, top_S, top_Vt


def project_embeddings_to_spin(embeds: torch.Tensor, spin_basis: torch.Tensor) -> torch.Tensor:
    """
    Project embeddings into SPIN space.
    
    This gives us coordinates in the "navigation" space.
    """
    # embeds: [vocab, hidden]
    # spin_basis: [hidden, n_dims]
    
    # Project
    coords = embeds @ spin_basis  # [vocab, n_dims]
    return coords


def analyze_semantic_pairs(coords: torch.Tensor, tokenizer, pairs: List[Tuple[str, str]]):
    """
    Analyze how semantic pairs behave in SPIN space.
    
    If SPIN encodes semantic dimensions:
    - Opposite pairs should differ primarily in ONE dimension
    - That dimension is the "semantic axis" for that concept
    """
    print(f"\n--- SEMANTIC PAIR ANALYSIS IN SPIN SPACE ---")
    
    results = []
    
    for neg_word, pos_word in pairs:
        neg_ids = tokenizer.encode(neg_word, add_special_tokens=False)
        pos_ids = tokenizer.encode(pos_word, add_special_tokens=False)
        
        if not neg_ids or not pos_ids:
            continue
        
        neg_coord = coords[neg_ids[0]]
        pos_coord = coords[pos_ids[0]]
        
        # Difference vector
        diff = pos_coord - neg_coord
        
        # Which dimension has the largest difference?
        max_dim = diff.abs().argmax().item()
        max_diff = diff[max_dim].item()
        
        # How concentrated is the difference? (is it mostly in one dimension?)
        diff_magnitude = diff.abs()
        total_diff = diff_magnitude.sum().item()
        concentration = diff_magnitude[max_dim].item() / total_diff if total_diff > 0 else 0
        
        results.append({
            'pair': (neg_word, pos_word),
            'max_dim': max_dim,
            'max_diff': max_diff,
            'concentration': concentration,
            'diff_vector': diff[:10].tolist(),  # First 10 dims
        })
        
        print(f"  {neg_word:12s} → {pos_word:12s}: dim={max_dim:2d}, diff={max_diff:+.4f}, conc={concentration:.1%}")
    
    # Group by dimension
    print(f"\n--- DIMENSION GROUPINGS ---")
    dim_groups = {}
    for r in results:
        d = r['max_dim']
        if d not in dim_groups:
            dim_groups[d] = []
        dim_groups[d].append(r['pair'])
    
    for dim, pairs_list in sorted(dim_groups.items(), key=lambda x: -len(x[1])):
        if len(pairs_list) >= 2:
            pairs_str = ", ".join([f"{p[0]}→{p[1]}" for p in pairs_list])
            print(f"  Dim {dim:2d}: {pairs_str}")
    
    return results


def analyze_mass_similarity(mass: torch.Tensor, embeds: torch.Tensor, tokenizer, words: List[str]):
    """
    Analyze how MASS encodes similarity.
    
    MASS is symmetric - it should encode "how similar" things are.
    """
    print(f"\n--- MASS SIMILARITY ANALYSIS ---")
    
    # Get embeddings for test words
    word_ids = []
    valid_words = []
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            word_ids.append(ids[0])
            valid_words.append(word)
    
    if len(word_ids) < 2:
        print("  Not enough valid words")
        return
    
    # Project through MASS
    word_embeds = embeds[word_ids]  # [n_words, hidden]
    
    # Similarity via MASS: sim[i,j] = embed_i @ MASS @ embed_j.T
    mass_sim = word_embeds @ mass @ word_embeds.T
    
    # Also compute direct cosine similarity for comparison
    word_embeds_norm = F.normalize(word_embeds, dim=1)
    cosine_sim = word_embeds_norm @ word_embeds_norm.T
    
    print(f"  MASS similarity matrix:")
    print(f"  {'':12s}", end="")
    for w in valid_words:
        print(f"{w:12s}", end="")
    print()
    
    for i, w1 in enumerate(valid_words):
        print(f"  {w1:12s}", end="")
        for j, w2 in enumerate(valid_words):
            print(f"{mass_sim[i,j].item():12.4f}", end="")
        print()
    
    print(f"\n  Cosine similarity matrix (for comparison):")
    print(f"  {'':12s}", end="")
    for w in valid_words:
        print(f"{w:12s}", end="")
    print()
    
    for i, w1 in enumerate(valid_words):
        print(f"  {w1:12s}", end="")
        for j, w2 in enumerate(valid_words):
            print(f"{cosine_sim[i,j].item():12.4f}", end="")
        print()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("="*70)
    print("EXTRACTING SEMANTIC DIMENSIONS FROM MESH")
    print("="*70)
    print("""
Semantic dimensions are tricky - they're not like other dimensions.

Hypothesis:
- SPIN (antisymmetric part of MESH) encodes navigation directions
- SPIN eigenvectors might BE the semantic dimensions
- Opposite pairs should differ primarily in ONE spin dimension
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
    
    # Known semantic pairs (our training data)
    semantic_pairs = [
        # Temperature
        ("cold", "hot"), ("cool", "warm"), ("freezing", "burning"),
        # Size
        ("small", "big"), ("tiny", "huge"), ("little", "large"),
        # Speed
        ("slow", "fast"), ("sluggish", "quick"), ("leisurely", "swift"),
        # Height
        ("short", "tall"), ("low", "high"),
        # Brightness
        ("dark", "bright"), ("dim", "light"),
        # Age
        ("young", "old"), ("new", "ancient"),
        # Valence
        ("bad", "good"), ("sad", "happy"), ("negative", "positive"),
        # Weight
        ("light", "heavy"),
        # Hardness
        ("soft", "hard"), ("gentle", "harsh"),
        # Moisture
        ("dry", "wet"),
    ]
    
    # Test words for similarity analysis
    test_words = ["hot", "cold", "warm", "cool", "big", "small", "happy", "sad", "love", "hate"]
    
    # Analyze multiple layers
    for layer_idx in [0, 14, 27]:
        print(f"\n{'='*70}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*70}")
        
        # Extract MESH
        mesh = extract_mesh(model, layer_idx)
        print(f"MESH shape: {mesh.shape}")
        
        # Decompose
        mass, spin = decompose_mesh(mesh)
        
        mass_energy = (mass ** 2).sum().item()
        spin_energy = (spin ** 2).sum().item()
        total = mass_energy + spin_energy
        print(f"MASS energy: {mass_energy/total*100:.1f}%")
        print(f"SPIN energy: {spin_energy/total*100:.1f}%")
        
        # Extract SPIN dimensions
        spin_U, spin_S, spin_Vt = extract_spin_dimensions(spin, n_dims=20)
        
        print(f"\nSPIN singular values (top 10):")
        for i in range(min(10, len(spin_S))):
            print(f"  S_{i}: {spin_S[i].item():.6f}")
        
        # Project embeddings to SPIN space
        spin_coords = project_embeddings_to_spin(embeds, spin_U)
        print(f"SPIN coordinates shape: {spin_coords.shape}")
        
        # Analyze semantic pairs in SPIN space
        analyze_semantic_pairs(spin_coords, tokenizer, semantic_pairs)
        
        # Analyze MASS similarity
        analyze_mass_similarity(mass, embeds, tokenizer, test_words)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
If semantic dimensions are encoded in SPIN:
- Opposite pairs should cluster by dimension
- Each dimension should capture one semantic axis
- The concentration should be high (>50%)

If MASS encodes similarity:
- Similar words should have high MASS similarity
- Opposite words should have low/negative MASS similarity
""")


if __name__ == "__main__":
    main()
