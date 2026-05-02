#!/usr/bin/env python3
"""
φ-Lattice Suspected Rules Verification
=======================================

Test the 5 remaining suspected rules:
16. Sign-based orthogonality
17. Semantic φ-spacing (analogies)
18. Transition selection rules
19. Q-O duality
20. V as identity transform
"""

import torch
import numpy as np
import math
from collections import Counter, defaultdict
from safetensors.torch import load_file
from pathlib import Path
from transformers import AutoTokenizer

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


def load_embeddings():
    """Load embedding weights."""
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_dirs = list(cache_dir.glob("models--Qwen--Qwen2-7B-Instruct/snapshots/*"))
    model_path = model_dirs[0]
    safetensor_files = list(model_path.glob("*.safetensors"))
    
    for sf_file in safetensor_files:
        tensors = load_file(sf_file)
        if "model.embed_tokens.weight" in tensors:
            embeddings = tensors["model.embed_tokens.weight"].float()
            del tensors
            return embeddings
    return None


def test_rule_16_sign_orthogonality():
    """Test if orthogonality is encoded in sign patterns."""
    print("="*70)
    print("RULE 16: SIGN-BASED ORTHOGONALITY")
    print("="*70)
    
    weight = load_attention_weights(14, "q_proj")
    n_heads = 28
    head_dim = 128
    hidden_dim = weight.shape[1]
    
    weight_heads = weight.reshape(n_heads, head_dim, hidden_dim)
    
    # Encode each head
    head_signs = []
    for h in range(n_heads):
        _, signs = encode_phi(weight_heads[h])
        head_signs.append(signs.flatten().float())
    
    # Compute sign-based similarity (like Hamming distance)
    print("\nSign agreement between heads:")
    sign_agreement = torch.zeros(n_heads, n_heads)
    for i in range(n_heads):
        for j in range(n_heads):
            # Fraction of matching signs
            agreement = (head_signs[i] == head_signs[j]).float().mean()
            sign_agreement[i, j] = agreement
    
    off_diag = sign_agreement[~torch.eye(n_heads, dtype=bool)]
    print(f"  Mean sign agreement (off-diagonal): {off_diag.mean():.4f}")
    print(f"  Expected if random: 0.5000")
    print(f"  Min agreement: {off_diag.min():.4f}")
    print(f"  Max agreement: {off_diag.max():.4f}")
    
    # If orthogonality is in signs, agreement should be ~0.5 (random)
    if abs(off_diag.mean() - 0.5) < 0.05:
        print("\n✓ RULE 16 VALIDATED: Signs are near-random between heads (orthogonal)")
    else:
        print(f"\n⚠ Signs show {off_diag.mean():.1%} agreement (not purely random)")
    
    return sign_agreement


def test_rule_17_semantic_spacing():
    """Test if semantic analogies have consistent φ-level differences."""
    print("\n" + "="*70)
    print("RULE 17: SEMANTIC φ-SPACING")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    embeddings = load_embeddings()
    
    # Encode all embeddings
    levels, signs = encode_phi(embeddings)
    
    # Test words (need to find their token IDs)
    test_pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("boy", "girl"),
        ("father", "mother"),
        ("brother", "sister"),
        ("uncle", "aunt"),
        ("prince", "princess"),
        ("actor", "actress"),
    ]
    
    print("\nSemantic pair level differences:")
    level_diffs = []
    
    for word1, word2 in test_pairs:
        # Get token IDs (use first token if multi-token)
        ids1 = tokenizer.encode(word1, add_special_tokens=False)
        ids2 = tokenizer.encode(word2, add_special_tokens=False)
        
        if ids1 and ids2:
            id1, id2 = ids1[0], ids2[0]
            
            # Get mean levels
            level1 = levels[id1].float().mean().item()
            level2 = levels[id2].float().mean().item()
            diff = level2 - level1
            level_diffs.append(diff)
            
            print(f"  {word1:10s} → {word2:10s}: Δlevel = {diff:+.1f}")
    
    if level_diffs:
        mean_diff = np.mean(level_diffs)
        std_diff = np.std(level_diffs)
        print(f"\n  Mean Δlevel: {mean_diff:+.1f}")
        print(f"  Std Δlevel: {std_diff:.1f}")
        
        # Check if differences are consistent
        if std_diff < abs(mean_diff) * 0.5:
            print("\n✓ RULE 17 VALIDATED: Semantic pairs have consistent level differences")
        else:
            print("\n⚠ Level differences vary significantly between pairs")
    
    return level_diffs


def test_rule_18_transition_selection():
    """Test if forbidden transitions follow a pattern."""
    print("\n" + "="*70)
    print("RULE 18: TRANSITION SELECTION RULES")
    print("="*70)
    
    weight = load_attention_weights(14, "q_proj")
    levels, _ = encode_phi(weight)
    levels = levels.flatten()
    
    # Compute all transitions
    diffs = (levels[1:] - levels[:-1]).numpy()
    diff_counts = Counter(diffs)
    
    # Analyze patterns in allowed vs forbidden
    all_diffs = sorted(diff_counts.keys())
    min_d, max_d = min(all_diffs), max(all_diffs)
    
    # Check parity pattern
    even_count = sum(c for d, c in diff_counts.items() if d % 2 == 0)
    odd_count = sum(c for d, c in diff_counts.items() if d % 2 == 1)
    total = even_count + odd_count
    
    print(f"\nParity analysis:")
    print(f"  Even transitions: {even_count/total*100:.1f}%")
    print(f"  Odd transitions: {odd_count/total*100:.1f}%")
    
    # Check φ-harmonic pattern (multiples of 128, 64, 32, etc.)
    print(f"\nφ-harmonic analysis:")
    for harmonic in [128, 64, 32, 16, 8, 4, 2]:
        near_harmonic = sum(c for d, c in diff_counts.items() 
                          if d != 0 and abs(d) % harmonic < harmonic * 0.1)
        print(f"  Near multiples of {harmonic}: {near_harmonic/total*100:.1f}%")
    
    # Check which transitions are forbidden
    forbidden = set(range(min_d, max_d + 1)) - set(diff_counts.keys())
    
    # Analyze forbidden transitions
    forbidden_even = len([d for d in forbidden if d % 2 == 0])
    forbidden_odd = len([d for d in forbidden if d % 2 == 1])
    
    print(f"\nForbidden transition analysis:")
    print(f"  Total forbidden: {len(forbidden)}")
    print(f"  Forbidden even: {forbidden_even} ({forbidden_even/len(forbidden)*100:.1f}%)")
    print(f"  Forbidden odd: {forbidden_odd} ({forbidden_odd/len(forbidden)*100:.1f}%)")
    
    # Check if large jumps are forbidden
    large_forbidden = len([d for d in forbidden if abs(d) > 1000])
    print(f"  Large (|Δ|>1000) forbidden: {large_forbidden}")


def test_rule_19_qo_duality():
    """Test if Q and O projections have complementary tetrominoes."""
    print("\n" + "="*70)
    print("RULE 19: Q-O DUALITY")
    print("="*70)
    
    # Load Q and O projections
    weight_q = load_attention_weights(14, "q_proj")
    weight_o = load_attention_weights(14, "o_proj")
    
    def get_tetrominoes(weight):
        weight = weight.flatten()
        n_blocks = len(weight) // 4
        W_4d = weight[:n_blocks*4].reshape(-1, 4)
        levels, signs = encode_phi(W_4d)
        block_levels = levels.float().mean(dim=1).round().to(torch.int32)
        
        tetrominoes = Counter()
        for i in range(len(W_4d)):
            bl = block_levels[i].item()
            sp = tuple(signs[i].tolist())
            tetrominoes[(bl, sp)] += 1
        return tetrominoes
    
    tet_q = get_tetrominoes(weight_q)
    tet_o = get_tetrominoes(weight_o)
    
    # Find unique to each
    unique_q = set(tet_q.keys()) - set(tet_o.keys())
    unique_o = set(tet_o.keys()) - set(tet_q.keys())
    shared = set(tet_q.keys()) & set(tet_o.keys())
    
    print(f"\nTetromino overlap:")
    print(f"  Unique to Q: {len(unique_q)}")
    print(f"  Unique to O: {len(unique_o)}")
    print(f"  Shared: {len(shared)}")
    
    # Check if Q's common are O's rare and vice versa
    q_top = set(t for t, _ in tet_q.most_common(100))
    o_top = set(t for t, _ in tet_o.most_common(100))
    
    q_top_in_o_top = len(q_top & o_top)
    print(f"\n  Q's top 100 that are also O's top 100: {q_top_in_o_top}")
    
    # Correlation of frequencies
    shared_list = list(shared)
    if shared_list:
        q_freqs = torch.tensor([tet_q[t] for t in shared_list], dtype=torch.float32)
        o_freqs = torch.tensor([tet_o[t] for t in shared_list], dtype=torch.float32)
        
        corr = torch.corrcoef(torch.stack([q_freqs, o_freqs]))[0, 1].item()
        print(f"\n  Frequency correlation (shared tetrominoes): {corr:.4f}")
        
        if corr < 0:
            print("\n✓ RULE 19 VALIDATED: Q and O have anti-correlated frequencies (duality)")
        elif corr > 0.8:
            print("\n⚠ Q and O have similar frequency patterns (not dual)")
        else:
            print("\n⚠ Q and O have weak correlation (partial duality)")


def test_rule_20_v_identity():
    """Test if V projection is close to identity."""
    print("\n" + "="*70)
    print("RULE 20: V AS IDENTITY TRANSFORM")
    print("="*70)
    
    # Load all projections
    projs = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    def get_tetromino_entropy(weight):
        """Compute entropy of tetromino distribution."""
        weight = weight.flatten()
        n_blocks = len(weight) // 4
        W_4d = weight[:n_blocks*4].reshape(-1, 4)
        levels, signs = encode_phi(W_4d)
        block_levels = levels.float().mean(dim=1).round().to(torch.int32)
        
        tetrominoes = Counter()
        for i in range(len(W_4d)):
            bl = block_levels[i].item()
            sp = tuple(signs[i].tolist())
            tetrominoes[(bl, sp)] += 1
        
        # Compute entropy
        total = sum(tetrominoes.values())
        probs = [c/total for c in tetrominoes.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        return entropy, len(tetrominoes)
    
    print("\nTetromino distribution analysis:")
    for proj in projs:
        weight = load_attention_weights(14, proj)
        entropy, n_unique = get_tetromino_entropy(weight)
        print(f"  {proj}: entropy = {entropy:.2f} bits, unique = {n_unique}")
    
    # Compare V to random baseline
    print("\n  Random baseline entropy: ~log2(n_unique) if uniform")
    
    # Check if V has highest entropy (most uniform = most generic)
    entropies = {}
    for proj in projs:
        weight = load_attention_weights(14, proj)
        entropy, _ = get_tetromino_entropy(weight)
        entropies[proj] = entropy
    
    if entropies['v_proj'] == max(entropies.values()):
        print("\n✓ RULE 20 VALIDATED: V has highest entropy (most generic/identity-like)")
    else:
        max_proj = max(entropies, key=entropies.get)
        print(f"\n⚠ {max_proj} has highest entropy, not V")


def main():
    print("="*70)
    print("φ-LATTICE SUSPECTED RULES VERIFICATION")
    print("="*70)
    
    test_rule_16_sign_orthogonality()
    test_rule_17_semantic_spacing()
    test_rule_18_transition_selection()
    test_rule_19_qo_duality()
    test_rule_20_v_identity()
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
