#!/usr/bin/env python3
"""
Qwen2.0 φ-Pattern Deep Analysis
================================

Deeper analysis looking for φ-patterns in:
1. Per-head attention patterns
2. Angle clustering across all layers
3. Singular value ratios
4. Weight correlations
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI

# φ-based reference angles
PHI_ANGLES = {
    'arctan_phi': np.arctan(PHI),           # 58.28°
    'arctan_phi_inv': np.arctan(PHI_INV),   # 31.72°
    'pi_over_phi': np.pi / PHI,             # 111.25°
    'pi_over_2phi': np.pi / (2 * PHI),      # 55.62°
    'phi_radians': PHI,                      # 92.73°
    'phi_inv_radians': PHI_INV,             # 35.42°
}


def load_model():
    """Load Qwen2-0.5B model."""
    print("Loading Qwen2-0.5B...")
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
        torch_dtype=torch.float16,
    )
    model = model.cpu()
    return model


def extract_attention_weights(model):
    """Extract all attention weights organized by layer."""
    weights = {}
    
    for name, param in model.named_parameters():
        if 'self_attn' in name and 'weight' in name:
            layer_idx = int(name.split('.')[2])
            proj_type = name.split('.')[-2]  # q_proj, k_proj, v_proj, o_proj
            
            if layer_idx not in weights:
                weights[layer_idx] = {}
            
            weights[layer_idx][proj_type] = param.detach().cpu().float().numpy()
    
    return weights


def analyze_per_head_patterns(W_q, W_k, head_dim=64):
    """
    Analyze attention patterns per head.
    
    For GQA, we need to handle the fact that multiple Q heads share K/V heads.
    W_q: [n_heads_q * head_dim, hidden_dim]
    W_k: [n_heads_kv * head_dim, hidden_dim]
    """
    n_heads_q = W_q.shape[0] // head_dim
    n_heads_kv = W_k.shape[0] // head_dim
    heads_per_group = n_heads_q // n_heads_kv
    
    results = []
    
    for kv_head in range(n_heads_kv):
        # Get K head weights
        k_start = kv_head * head_dim
        k_end = k_start + head_dim
        W_k_head = W_k[k_start:k_end, :]  # [head_dim, hidden_dim]
        
        for q_offset in range(heads_per_group):
            q_head = kv_head * heads_per_group + q_offset
            q_start = q_head * head_dim
            q_end = q_start + head_dim
            W_q_head = W_q[q_start:q_end, :]  # [head_dim, hidden_dim]
            
            # Compute per-head MESH
            # MESH = W_q_head @ W_k_head.T gives [head_dim, head_dim]
            MESH_head = W_q_head @ W_k_head.T
            
            # SVD of per-head MESH
            U, S, Vt = np.linalg.svd(MESH_head, full_matrices=False)
            
            # Compute angles between rows of MESH
            norms = np.linalg.norm(MESH_head, axis=1, keepdims=True)
            MESH_norm = MESH_head / (norms + 1e-10)
            
            # All pairwise angles
            angles = []
            for i in range(head_dim):
                for j in range(i + 1, head_dim):
                    dot = np.clip(np.dot(MESH_norm[i], MESH_norm[j]), -1, 1)
                    angle = np.arccos(dot)
                    angles.append(angle)
            
            angles = np.array(angles)
            
            results.append({
                'q_head': q_head,
                'kv_head': kv_head,
                'singular_values': S,
                'sv_ratio_01': S[0] / S[1] if len(S) > 1 else 0,
                'angles_mean': np.mean(angles),
                'angles_std': np.std(angles),
                'angles_min': np.min(angles),
                'angles_max': np.max(angles),
            })
    
    return results


def find_phi_angles(angles, tolerance=0.05):
    """Find angles that match φ-based reference angles."""
    matches = []
    
    for angle in angles:
        for name, ref_angle in PHI_ANGLES.items():
            if abs(angle - ref_angle) < tolerance:
                matches.append({
                    'angle': angle,
                    'angle_deg': np.degrees(angle),
                    'reference': name,
                    'ref_angle_deg': np.degrees(ref_angle),
                    'error': abs(angle - ref_angle)
                })
    
    return matches


def analyze_singular_value_ratios(weights_by_layer):
    """Analyze singular value ratios across all layers."""
    print()
    print("=" * 70)
    print("SINGULAR VALUE RATIO ANALYSIS ACROSS LAYERS")
    print("=" * 70)
    print()
    
    all_ratios = []
    
    for layer_idx in sorted(weights_by_layer.keys()):
        layer_weights = weights_by_layer[layer_idx]
        
        for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if proj_type not in layer_weights:
                continue
            
            W = layer_weights[proj_type]
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            
            # Compute consecutive ratios
            ratios = S[:-1] / S[1:]
            
            # Look for φ-ratios
            for i, r in enumerate(ratios[:10]):
                phi_diff = abs(r - PHI)
                phi_inv_diff = abs(r - PHI_INV)
                
                if phi_diff < 0.1 or phi_inv_diff < 0.1:
                    match_type = 'φ' if phi_diff < phi_inv_diff else '1/φ'
                    all_ratios.append({
                        'layer': layer_idx,
                        'proj': proj_type,
                        'index': i,
                        'ratio': r,
                        'match': match_type,
                        'error': min(phi_diff, phi_inv_diff)
                    })
    
    print(f"Found {len(all_ratios)} φ-ratio matches across all layers")
    
    if all_ratios:
        print()
        print("Sample matches:")
        for m in all_ratios[:20]:
            print(f"  Layer {m['layer']:2d} {m['proj']:6s} S[{m['index']}]/S[{m['index']+1}] = {m['ratio']:.4f} ≈ {m['match']}")
    
    return all_ratios


def analyze_angle_clustering(weights_by_layer):
    """Analyze angle clustering across all layers."""
    print()
    print("=" * 70)
    print("ANGLE CLUSTERING ANALYSIS")
    print("=" * 70)
    print()
    
    all_angles = []
    phi_matches = []
    
    for layer_idx in sorted(weights_by_layer.keys())[:5]:  # First 5 layers for speed
        layer_weights = weights_by_layer[layer_idx]
        
        W_q = layer_weights.get('q_proj')
        W_k = layer_weights.get('k_proj')
        
        if W_q is None or W_k is None:
            continue
        
        print(f"Analyzing Layer {layer_idx}...")
        
        # Per-head analysis
        head_results = analyze_per_head_patterns(W_q, W_k)
        
        for hr in head_results:
            all_angles.append(hr['angles_mean'])
            
            # Check if mean angle matches φ-angles
            matches = find_phi_angles([hr['angles_mean']])
            if matches:
                for m in matches:
                    m['layer'] = layer_idx
                    m['q_head'] = hr['q_head']
                    phi_matches.append(m)
    
    all_angles = np.array(all_angles)
    
    print()
    print(f"Analyzed {len(all_angles)} head patterns")
    print(f"Mean angle: {np.degrees(np.mean(all_angles)):.2f}°")
    print(f"Std angle: {np.degrees(np.std(all_angles)):.2f}°")
    
    # Histogram of angles
    print()
    print("Angle distribution (degrees):")
    hist, bins = np.histogram(np.degrees(all_angles), bins=18, range=(0, 180))
    for i, count in enumerate(hist):
        if count > 0:
            bin_center = (bins[i] + bins[i+1]) / 2
            bar = '#' * count
            print(f"  {bin_center:5.1f}°: {bar} ({count})")
    
    if phi_matches:
        print()
        print(f"Found {len(phi_matches)} φ-angle matches:")
        for m in phi_matches[:10]:
            print(f"  Layer {m['layer']} Head {m['q_head']}: {m['angle_deg']:.2f}° ≈ {m['reference']} ({m['ref_angle_deg']:.2f}°)")
    
    return all_angles, phi_matches


def analyze_weight_correlations(weights_by_layer):
    """Analyze correlations between weight matrices across layers."""
    print()
    print("=" * 70)
    print("WEIGHT CORRELATION ANALYSIS")
    print("=" * 70)
    print()
    
    # Compare Q projections across layers
    q_correlations = []
    
    layers = sorted(weights_by_layer.keys())
    
    for i in range(len(layers) - 1):
        W_q_i = weights_by_layer[layers[i]]['q_proj'].flatten()
        W_q_j = weights_by_layer[layers[i + 1]]['q_proj'].flatten()
        
        corr = np.corrcoef(W_q_i, W_q_j)[0, 1]
        q_correlations.append(corr)
    
    print("Q projection correlations between adjacent layers:")
    for i, corr in enumerate(q_correlations):
        print(f"  Layer {i} <-> Layer {i+1}: {corr:.4f}")
    
    print()
    print(f"Mean correlation: {np.mean(q_correlations):.4f}")
    print(f"Std correlation: {np.std(q_correlations):.4f}")
    
    # Check if correlations follow φ-pattern
    print()
    print("Checking for φ-patterns in correlation decay...")
    
    # Correlation with layer 0
    layer0_corrs = []
    W_q_0 = weights_by_layer[0]['q_proj'].flatten()
    
    for layer_idx in layers[1:]:
        W_q_i = weights_by_layer[layer_idx]['q_proj'].flatten()
        corr = np.corrcoef(W_q_0, W_q_i)[0, 1]
        layer0_corrs.append((layer_idx, corr))
    
    print("Correlation with Layer 0:")
    for layer_idx, corr in layer0_corrs[:10]:
        print(f"  Layer {layer_idx:2d}: {corr:.4f}")
    
    return q_correlations, layer0_corrs


def main():
    # Load model
    model = load_model()
    
    # Extract weights
    print()
    print("Extracting attention weights...")
    weights_by_layer = extract_attention_weights(model)
    print(f"Extracted weights from {len(weights_by_layer)} layers")
    
    # Analysis 1: Singular value ratios
    sv_ratios = analyze_singular_value_ratios(weights_by_layer)
    
    # Analysis 2: Angle clustering
    angles, phi_matches = analyze_angle_clustering(weights_by_layer)
    
    # Analysis 3: Weight correlations
    adj_corrs, layer0_corrs = analyze_weight_correlations(weights_by_layer)
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"φ-ratio matches in singular values: {len(sv_ratios)}")
    print(f"φ-angle matches in attention heads: {len(phi_matches)}")
    print(f"Mean adjacent layer correlation: {np.mean(adj_corrs):.4f}")
    
    # Save results
    results = {
        'sv_ratio_matches': len(sv_ratios),
        'phi_angle_matches': len(phi_matches),
        'mean_angle_deg': float(np.degrees(np.mean(angles))),
        'mean_adj_correlation': float(np.mean(adj_corrs)),
    }
    
    with open('qwen2_phi_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("Results saved to qwen2_phi_analysis_results.json")


if __name__ == "__main__":
    main()
