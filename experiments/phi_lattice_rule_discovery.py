#!/usr/bin/env python3
"""
φ-Lattice Rule Discovery
=========================

Test suspected rules and look for new patterns.
"""

import torch
import numpy as np
import math
from collections import Counter, defaultdict
from safetensors.torch import load_file
from pathlib import Path

PHI = (1 + math.sqrt(5)) / 2
LOG_PHI = math.log(PHI)
K_SCALE = 128


def encode_phi(tensor):
    """Encode tensor to φ-lattice."""
    signs = torch.sign(tensor)
    signs[signs == 0] = 1
    magnitudes = tensor.abs().clamp(min=1e-45)
    levels = torch.round(K_SCALE * torch.log(magnitudes) / LOG_PHI)
    return levels.to(torch.int32), signs.to(torch.int8)


def load_attention_weights(layer_idx, proj):
    """Load attention weights from safetensors."""
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_dirs = list(cache_dir.glob("models--Qwen--Qwen2-7B-Instruct/snapshots/*"))
    model_path = model_dirs[0]
    safetensor_files = list(model_path.glob("*.safetensors"))
    
    key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
    for sf_file in safetensor_files:
        tensors = load_file(sf_file)
        if key in tensors:
            weight = tensors[key].float()
            del tensors
            return weight
    return None


def test_orthogonality_in_phi_space():
    """Test if attention heads are orthogonal in φ-space."""
    print("="*70)
    print("TEST: ORTHOGONALITY IN φ-SPACE")
    print("="*70)
    
    # Load Q projection from layer 14
    weight = load_attention_weights(14, "q_proj")
    print(f"Weight shape: {weight.shape}")
    
    # Qwen2-7B has 28 heads, each with dim 128
    n_heads = 28
    head_dim = 128
    hidden_dim = weight.shape[1]
    
    # Reshape to [n_heads, head_dim, hidden_dim]
    weight_heads = weight.reshape(n_heads, head_dim, hidden_dim)
    
    # Encode each head to φ-space
    head_levels = []
    head_signs = []
    for h in range(n_heads):
        levels, signs = encode_phi(weight_heads[h])
        head_levels.append(levels.flatten().float())
        head_signs.append(signs.flatten().float())
    
    # Compute pairwise dot products in level-space
    print("\nDot products between heads (level-space):")
    level_dots = torch.zeros(n_heads, n_heads)
    for i in range(n_heads):
        for j in range(n_heads):
            # Normalize
            li = head_levels[i] / (head_levels[i].norm() + 1e-10)
            lj = head_levels[j] / (head_levels[j].norm() + 1e-10)
            level_dots[i, j] = (li * lj).sum()
    
    # Check orthogonality
    off_diag = level_dots[~torch.eye(n_heads, dtype=bool)]
    print(f"  Mean off-diagonal dot product: {off_diag.mean():.4f}")
    print(f"  Max off-diagonal dot product: {off_diag.abs().max():.4f}")
    print(f"  Std of off-diagonal: {off_diag.std():.4f}")
    
    # Compare to original space
    print("\nDot products between heads (original space):")
    orig_dots = torch.zeros(n_heads, n_heads)
    for i in range(n_heads):
        for j in range(n_heads):
            wi = weight_heads[i].flatten()
            wj = weight_heads[j].flatten()
            wi = wi / (wi.norm() + 1e-10)
            wj = wj / (wj.norm() + 1e-10)
            orig_dots[i, j] = (wi * wj).sum()
    
    off_diag_orig = orig_dots[~torch.eye(n_heads, dtype=bool)]
    print(f"  Mean off-diagonal dot product: {off_diag_orig.mean():.4f}")
    print(f"  Max off-diagonal dot product: {off_diag_orig.abs().max():.4f}")
    
    # Conclusion
    if off_diag.abs().max() < 0.3:
        print("\n✓ RULE VALIDATED: Heads are near-orthogonal in φ-space")
    else:
        print("\n⚠ RULE NEEDS REFINEMENT: Heads show some correlation in φ-space")


def test_phi_ratio_relationships():
    """Test if semantic relationships have φ-ratio level differences."""
    print("\n" + "="*70)
    print("TEST: φ-RATIO RELATIONSHIPS")
    print("="*70)
    
    # Load embedding layer
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_dirs = list(cache_dir.glob("models--Qwen--Qwen2-7B-Instruct/snapshots/*"))
    model_path = model_dirs[0]
    safetensor_files = list(model_path.glob("*.safetensors"))
    
    for sf_file in safetensor_files:
        tensors = load_file(sf_file)
        if "model.embed_tokens.weight" in tensors:
            embeddings = tensors["model.embed_tokens.weight"].float()
            del tensors
            break
    
    print(f"Embedding shape: {embeddings.shape}")
    
    # Encode all embeddings to φ-space
    levels, signs = encode_phi(embeddings)
    
    # Compute mean level per token
    token_mean_levels = levels.float().mean(dim=1)
    
    print(f"\nToken level statistics:")
    print(f"  Mean: {token_mean_levels.mean():.1f}")
    print(f"  Std: {token_mean_levels.std():.1f}")
    print(f"  Min: {token_mean_levels.min():.1f}")
    print(f"  Max: {token_mean_levels.max():.1f}")
    
    # Look for φ-ratio spacing
    # If levels are φ-spaced, differences should cluster at multiples of K (128)
    # because φ^1 in K=128 space is level difference of 128
    
    # Sample random pairs and compute level differences
    n_samples = 10000
    idx1 = torch.randint(0, len(token_mean_levels), (n_samples,))
    idx2 = torch.randint(0, len(token_mean_levels), (n_samples,))
    diffs = (token_mean_levels[idx1] - token_mean_levels[idx2]).abs()
    
    # Check if differences cluster at φ-harmonics
    # φ^0 = 1 → level diff 0
    # φ^1 = φ → level diff 128
    # φ^2 = φ+1 → level diff 256
    
    print("\nLevel difference distribution (looking for φ-harmonics):")
    for harmonic in [0, 64, 128, 192, 256]:
        # Count diffs within ±10 of harmonic
        near_harmonic = ((diffs - harmonic).abs() < 10).sum().item()
        pct = near_harmonic / n_samples * 100
        phi_power = harmonic / 128
        print(f"  Near {harmonic} (φ^{phi_power:.1f}): {pct:.1f}%")
    
    # Check if distribution is uniform or clustered
    print("\n  (If uniform, each bin would be ~2%)")


def test_conservation_laws():
    """Test if there are conserved quantities in the φ-lattice."""
    print("\n" + "="*70)
    print("TEST: CONSERVATION LAWS")
    print("="*70)
    
    # Load weights from multiple layers
    layers = [0, 7, 14, 21, 27]
    
    level_sums = []
    level_means = []
    sign_products = []
    level_variances = []
    
    for layer_idx in layers:
        weight = load_attention_weights(layer_idx, "q_proj")
        levels, signs = encode_phi(weight)
        
        level_sums.append(levels.float().sum().item())
        level_means.append(levels.float().mean().item())
        level_variances.append(levels.float().var().item())
        
        # Sign product (mod 2 - like parity)
        sign_prod = (signs == -1).sum().item() % 2
        sign_products.append(sign_prod)
    
    print("\nPotential conserved quantities across layers:")
    print(f"\n  Level sums: {[f'{x:.0f}' for x in level_sums]}")
    print(f"  Variance: {np.std(level_sums)/np.mean(np.abs(level_sums))*100:.1f}%")
    
    print(f"\n  Level means: {[f'{x:.1f}' for x in level_means]}")
    print(f"  Variance: {np.std(level_means):.2f}")
    
    print(f"\n  Level variances: {[f'{x:.0f}' for x in level_variances]}")
    print(f"  Variance: {np.std(level_variances)/np.mean(level_variances)*100:.1f}%")
    
    print(f"\n  Sign parities: {sign_products}")
    
    # Check which is most conserved
    sum_var = np.std(level_sums)/np.mean(np.abs(level_sums))
    mean_var = np.std(level_means)/np.mean(np.abs(level_means))
    var_var = np.std(level_variances)/np.mean(level_variances)
    
    print("\n  Most conserved quantity:", end=" ")
    if mean_var < sum_var and mean_var < var_var:
        print("LEVEL MEAN (like temperature)")
    elif var_var < sum_var:
        print("LEVEL VARIANCE (like entropy)")
    else:
        print("LEVEL SUM (like energy)")


def test_forbidden_transitions():
    """Test if certain level transitions are forbidden."""
    print("\n" + "="*70)
    print("TEST: FORBIDDEN TRANSITIONS")
    print("="*70)
    
    # Load Q projection and analyze internal level transitions
    weight_q = load_attention_weights(14, "q_proj")
    
    levels_q, _ = encode_phi(weight_q)
    levels_q = levels_q.flatten()
    
    # Compute level differences between adjacent weights
    # (This shows what level "jumps" occur in the weight matrix)
    diffs = (levels_q[1:] - levels_q[:-1]).numpy()
    
    # Count transition frequencies
    diff_counts = Counter(diffs)
    
    print("\nQ-K level differences (proxy for allowed transitions):")
    print("\nMost common transitions:")
    for diff, count in diff_counts.most_common(10):
        pct = count / len(diffs) * 100
        print(f"  Δ = {diff:+4d}: {pct:.2f}%")
    
    print("\nLeast common transitions (potential forbidden):")
    # Find gaps in the distribution
    all_diffs = sorted(diff_counts.keys())
    min_diff, max_diff = min(all_diffs), max(all_diffs)
    
    # Look for missing values in the range
    missing = []
    for d in range(min_diff, max_diff + 1):
        if d not in diff_counts:
            missing.append(d)
    
    if missing:
        print(f"  Missing transitions: {missing[:20]}...")
        print(f"  Total missing: {len(missing)} out of {max_diff - min_diff + 1} possible")
    else:
        print("  No completely forbidden transitions found")
    
    # Check for suppressed transitions (very rare)
    threshold = len(diffs) * 0.0001  # 0.01%
    suppressed = [d for d, c in diff_counts.items() if c < threshold]
    print(f"\n  Suppressed transitions (<0.01%): {len(suppressed)}")


def test_tetromino_semantics():
    """Explore if the 300 tetrominoes have semantic meaning."""
    print("\n" + "="*70)
    print("TEST: TETROMINO SEMANTICS")
    print("="*70)
    
    # Analyze tetromino distribution across different projections
    projs = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    tetromino_by_proj = defaultdict(Counter)
    
    for proj in projs:
        weight = load_attention_weights(14, proj)
        weight = weight.flatten()
        
        # Reshape to 4D blocks
        n_blocks = len(weight) // 4
        W_4d = weight[:n_blocks*4].reshape(-1, 4)
        
        levels, signs = encode_phi(W_4d)
        block_levels = levels.float().mean(dim=1).round().to(torch.int32)
        
        # Create tetromino signatures
        for i in range(len(W_4d)):
            bl = block_levels[i].item()
            sp = tuple(signs[i].tolist())
            tetromino_by_proj[proj][(bl, sp)] += 1
    
    print("\nTop tetrominoes by projection:")
    for proj in projs:
        print(f"\n{proj}:")
        for (bl, sp), count in tetromino_by_proj[proj].most_common(5):
            sp_str = "".join("+" if s > 0 else "-" for s in sp)
            pct = count / sum(tetromino_by_proj[proj].values()) * 100
            print(f"  φ^{bl/K_SCALE:.2f} × [{sp_str}]: {pct:.2f}%")
    
    # Check if projections have different tetromino preferences
    print("\nProjection-specific tetrominoes:")
    all_tetrominoes = set()
    for proj in projs:
        all_tetrominoes.update(tetromino_by_proj[proj].keys())
    
    for proj in projs:
        unique_to_proj = 0
        for t in tetromino_by_proj[proj]:
            # Check if this tetromino is much more common in this proj
            this_count = tetromino_by_proj[proj][t]
            other_counts = [tetromino_by_proj[p][t] for p in projs if p != proj]
            if this_count > 2 * max(other_counts + [1]):
                unique_to_proj += 1
        print(f"  {proj}: {unique_to_proj} projection-specific tetrominoes")


def main():
    print("="*70)
    print("φ-LATTICE RULE DISCOVERY")
    print("="*70)
    print("\nTesting suspected rules and looking for new patterns...\n")
    
    test_orthogonality_in_phi_space()
    test_phi_ratio_relationships()
    test_conservation_laws()
    test_forbidden_transitions()
    test_tetromino_semantics()
    
    print("\n" + "="*70)
    print("DISCOVERY COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
