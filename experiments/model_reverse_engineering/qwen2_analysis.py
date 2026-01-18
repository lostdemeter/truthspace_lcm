#!/usr/bin/env python3
"""
Qwen2.0 Architecture Analysis
==============================

Reverse engineering Qwen2.0 to understand its geometric structure
and map it to φ-basis representation.

This script:
1. Loads Qwen2.0 and enumerates all components
2. Analyzes weight matrices for φ-patterns
3. Extracts attention patterns and looks for φ-angles
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


def analyze_model_architecture(model):
    """Enumerate all modules and their shapes."""
    print("=" * 70)
    print("QWEN2.0 ARCHITECTURE ANALYSIS")
    print("=" * 70)
    print()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print()
    
    # Enumerate modules
    print("Module Structure:")
    print("-" * 70)
    
    module_types = defaultdict(list)
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_type = type(module).__name__
            module_types[module_type].append(name)
            
            # Get shape info
            shapes = []
            for pname, param in module.named_parameters(recurse=False):
                shapes.append(f"{pname}: {list(param.shape)}")
            
            if shapes:
                print(f"  {name}")
                print(f"    Type: {module_type}")
                for s in shapes:
                    print(f"    {s}")
    
    print()
    print("Module Type Summary:")
    print("-" * 70)
    for mtype, names in sorted(module_types.items(), key=lambda x: -len(x[1])):
        print(f"  {mtype}: {len(names)} instances")
    
    return module_types


def analyze_layer_structure(model):
    """Analyze the structure of transformer layers."""
    print()
    print("=" * 70)
    print("TRANSFORMER LAYER STRUCTURE")
    print("=" * 70)
    print()
    
    # Find the layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        print("Could not find transformer layers")
        return None
    
    print(f"Number of layers: {len(layers)}")
    print()
    
    # Analyze first layer in detail
    layer0 = layers[0]
    print("Layer 0 structure:")
    print("-" * 70)
    
    for name, module in layer0.named_modules():
        if len(list(module.children())) == 0:
            params = list(module.named_parameters(recurse=False))
            if params:
                print(f"  {name} ({type(module).__name__})")
                for pname, param in params:
                    print(f"    {pname}: {list(param.shape)}")
    
    return layers


def analyze_attention_weights(model):
    """Extract and analyze attention weight matrices."""
    print()
    print("=" * 70)
    print("ATTENTION WEIGHT ANALYSIS")
    print("=" * 70)
    print()
    
    # Find attention modules
    attention_weights = {}
    
    for name, param in model.named_parameters():
        if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name:
            attention_weights[name] = param.detach().cpu()
            print(f"Found: {name} {list(param.shape)}")
    
    print()
    print(f"Total attention weight matrices: {len(attention_weights)}")
    
    return attention_weights


def analyze_phi_patterns_in_weights(weights_dict, layer_idx=0):
    """Look for φ-patterns in weight matrices."""
    print()
    print("=" * 70)
    print(f"φ-PATTERN ANALYSIS (Layer {layer_idx})")
    print("=" * 70)
    print()
    
    # Get Q and K weights for the specified layer
    q_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
    k_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
    v_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
    o_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
    
    if q_key not in weights_dict or k_key not in weights_dict:
        print(f"Could not find Q/K weights for layer {layer_idx}")
        print(f"Available keys: {list(weights_dict.keys())[:5]}...")
        return None
    
    W_q = weights_dict[q_key].float().numpy()
    W_k = weights_dict[k_key].float().numpy()
    W_v = weights_dict[v_key].float().numpy()
    W_o = weights_dict[o_key].float().numpy()
    
    print(f"W_q shape: {W_q.shape}")
    print(f"W_k shape: {W_k.shape}")
    print(f"W_v shape: {W_v.shape}")
    print(f"W_o shape: {W_o.shape}")
    
    # Qwen2 uses Grouped Query Attention (GQA)
    # Q has more heads than K/V (K/V are shared across groups)
    n_heads_q = W_q.shape[0] // (W_k.shape[0] // (W_k.shape[0] // 128)) if W_k.shape[0] != W_q.shape[0] else W_q.shape[0] // 64
    n_heads_kv = W_k.shape[0] // 64 if W_k.shape[0] < W_q.shape[0] else n_heads_q
    head_dim = W_q.shape[1] // (W_q.shape[0] // 64) if W_q.shape[0] > 64 else 64
    
    print()
    print("Grouped Query Attention (GQA) detected:")
    print(f"  Q output dim: {W_q.shape[0]}")
    print(f"  K output dim: {W_k.shape[0]}")
    print(f"  Ratio (Q/K): {W_q.shape[0] / W_k.shape[0]:.1f}x")
    print(f"  This means {W_q.shape[0] // W_k.shape[0]} Q heads share each K/V head")
    
    # For GQA, we analyze each head separately
    # First, let's analyze the Q projection matrix itself
    print()
    print("Analyzing Q projection matrix...")
    
    # SVD of W_q
    U_q, S_q, Vt_q = np.linalg.svd(W_q, full_matrices=False)
    print(f"Q singular values (top 20): {S_q[:20].round(4)}")
    
    # SVD of W_k  
    U_k, S_k, Vt_k = np.linalg.svd(W_k, full_matrices=False)
    print(f"K singular values (top 20): {S_k[:20].round(4)}")
    
    # Compute MESH within the shared dimension space
    # W_q: (896, 896), W_k: (128, 896)
    # MESH = W_k @ W_q.T gives us (128, 896) - how K relates to Q
    print()
    print("Computing MESH = W_k @ W_q.T...")
    MESH = W_k @ W_q.T
    print(f"MESH shape: {MESH.shape}")
    
    # SVD of MESH
    print()
    print("SVD of MESH...")
    U, S, Vt = np.linalg.svd(MESH, full_matrices=False)
    
    print(f"Singular values (top 20): {S[:20]}")
    print(f"Singular value ratio (S[0]/S[1]): {S[0]/S[1]:.4f}")
    print(f"φ = {PHI:.4f}, 1/φ = {PHI_INV:.4f}")
    
    # Check for φ-ratios in singular values
    print()
    print("Checking for φ-ratios in singular values...")
    ratios = S[:-1] / S[1:]
    phi_matches = []
    for i, r in enumerate(ratios[:20]):
        phi_diff = abs(r - PHI)
        phi_inv_diff = abs(r - PHI_INV)
        if phi_diff < 0.1 or phi_inv_diff < 0.1:
            phi_matches.append((i, r, 'φ' if phi_diff < phi_inv_diff else '1/φ'))
            print(f"  S[{i}]/S[{i+1}] = {r:.4f} ≈ {'φ' if phi_diff < phi_inv_diff else '1/φ'}")
    
    # Analyze angles in MESH
    print()
    print("Analyzing angles in MESH...")
    
    # Normalize rows and compute angles
    norms = np.linalg.norm(MESH, axis=1, keepdims=True)
    MESH_norm = MESH / (norms + 1e-10)
    
    # Sample pairwise angles
    n_samples = min(1000, MESH.shape[0])
    indices = np.random.choice(MESH.shape[0], n_samples, replace=False)
    
    angles = []
    for i in range(0, len(indices), 2):
        if i + 1 < len(indices):
            dot = np.clip(np.dot(MESH_norm[indices[i]], MESH_norm[indices[i+1]]), -1, 1)
            angle = np.arccos(dot)
            angles.append(angle)
    
    angles = np.array(angles)
    print(f"Sampled {len(angles)} pairwise angles")
    print(f"Angle range: [{angles.min():.4f}, {angles.max():.4f}] radians")
    print(f"Angle range: [{np.degrees(angles.min()):.1f}°, {np.degrees(angles.max()):.1f}°]")
    
    # Check for φ-based angles
    phi_angles = [
        np.arctan(PHI),      # ~58.28°
        np.arctan(1/PHI),    # ~31.72°
        np.pi / PHI,         # ~114.09°
        np.pi / (PHI * 2),   # ~57.05°
    ]
    
    print()
    print("φ-based reference angles:")
    for pa in phi_angles:
        print(f"  {np.degrees(pa):.2f}°")
    
    return {
        'MESH': MESH,
        'singular_values': S,
        'U': U,
        'Vt': Vt,
        'angles': angles,
        'phi_matches': phi_matches
    }


def analyze_mlp_structure(model, layer_idx=0):
    """Analyze MLP (feed-forward) structure."""
    print()
    print("=" * 70)
    print(f"MLP STRUCTURE ANALYSIS (Layer {layer_idx})")
    print("=" * 70)
    print()
    
    mlp_weights = {}
    
    for name, param in model.named_parameters():
        if f'layers.{layer_idx}.mlp' in name:
            mlp_weights[name] = param.detach().cpu()
            print(f"Found: {name} {list(param.shape)}")
    
    # Qwen2 uses SwiGLU: gate_proj, up_proj, down_proj
    gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
    up_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
    down_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
    
    if gate_key in mlp_weights:
        W_gate = mlp_weights[gate_key].float().numpy()
        W_up = mlp_weights[up_key].float().numpy()
        W_down = mlp_weights[down_key].float().numpy()
        
        print()
        print("MLP dimensions:")
        print(f"  gate_proj: {W_gate.shape} (hidden -> intermediate)")
        print(f"  up_proj: {W_up.shape} (hidden -> intermediate)")
        print(f"  down_proj: {W_down.shape} (intermediate -> hidden)")
        
        # Check expansion ratio
        hidden_dim = W_gate.shape[1]
        intermediate_dim = W_gate.shape[0]
        ratio = intermediate_dim / hidden_dim
        print()
        print(f"Expansion ratio: {ratio:.4f}")
        print(f"φ² = {PHI**2:.4f}")
        print(f"8/3 = {8/3:.4f}")
        
        # SVD of gate projection
        print()
        print("SVD of gate_proj...")
        U, S, Vt = np.linalg.svd(W_gate, full_matrices=False)
        print(f"Top 10 singular values: {S[:10]}")
        
        return {
            'gate': W_gate,
            'up': W_up,
            'down': W_down,
            'expansion_ratio': ratio
        }
    
    return None


def main():
    print("Loading Qwen2-0.5B...")
    print("(Using smallest variant for faster analysis)")
    print()
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "Qwen/Qwen2-0.5B"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model = model.cpu()  # Move to CPU for analysis
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Model loaded: {model_name}")
    print()
    
    # Phase 1: Architecture analysis
    module_types = analyze_model_architecture(model)
    
    # Phase 2: Layer structure
    layers = analyze_layer_structure(model)
    
    # Phase 3: Attention weights
    attention_weights = analyze_attention_weights(model)
    
    # Phase 4: φ-pattern analysis
    phi_analysis = analyze_phi_patterns_in_weights(attention_weights, layer_idx=0)
    
    # Phase 5: MLP analysis
    mlp_analysis = analyze_mlp_structure(model, layer_idx=0)
    
    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Analyze φ-patterns across all layers")
    print("2. Look for consistent angles in attention")
    print("3. Map to φ-basis representation")
    print("4. Implement AIG-optimized version")


if __name__ == "__main__":
    main()
