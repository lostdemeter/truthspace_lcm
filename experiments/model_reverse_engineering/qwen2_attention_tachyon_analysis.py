#!/usr/bin/env python3
"""
Qwen2.0 Attention as Tachyon Navigation
=========================================

Hypothesis: Attention in sequences implements W-axis (tachyon) navigation.

From DA2 (doc 124):
- Q and K are 90° rotated (orthogonal)
- MESH = W_q.T @ W_k = MASS + SPIN
- MASS = symmetric (similarity)
- SPIN = antisymmetric (navigation)

From doc 055:
- W-axis = tachyon navigation direction
- φ^+n = forward attention (past → present)
- φ^-n = backward attention (future → present)
- Verbs are tachyon joints

Question: Does Qwen2's attention follow this pattern?
- Is Q·K orthogonal (90° rotation)?
- Does MESH decompose into MASS + SPIN?
- Does SPIN relate to φ-structure?
"""

import torch
import numpy as np
from pathlib import Path

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


def load_model():
    """Load Qwen2-0.5B model."""
    print("Loading Qwen2-0.5B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
        torch_dtype=torch.float32,
        attn_implementation="eager",  # Required for attention output
    )
    model = model.cpu()
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    
    return model, tokenizer


def analyze_qk_orthogonality(model):
    """
    Check if Q and K are orthogonal (90° rotated) like in DA2.
    
    For GQA, we have:
    - W_q: [896, 896] (14 heads × 64 dim)
    - W_k: [128, 896] (2 heads × 64 dim)
    
    The 7:1 ratio means 7 Q heads share each K head.
    """
    print()
    print("=" * 70)
    print("Q-K ORTHOGONALITY ANALYSIS")
    print("=" * 70)
    print()
    
    results = []
    
    for layer_idx in [0, 1, 2, 3, 11, 23]:  # Sample layers
        layer = model.model.layers[layer_idx]
        
        W_q = layer.self_attn.q_proj.weight.detach().cpu().float().numpy()
        W_k = layer.self_attn.k_proj.weight.detach().cpu().float().numpy()
        
        # For GQA, expand K to match Q dimensions
        # W_k is [128, 896], need to expand to [896, 896]
        # Each K head is used by 7 Q heads
        n_q_heads = 14
        n_kv_heads = 2
        head_dim = 64
        
        # Reshape
        W_q_heads = W_q.reshape(n_q_heads, head_dim, -1)  # [14, 64, 896]
        W_k_heads = W_k.reshape(n_kv_heads, head_dim, -1)  # [2, 64, 896]
        
        # Analyze per-head Q·K relationship
        for q_head in range(min(3, n_q_heads)):  # First 3 Q heads
            k_head = q_head // 7  # Which K head this Q uses
            
            W_q_h = W_q_heads[q_head]  # [64, 896]
            W_k_h = W_k_heads[k_head]  # [64, 896]
            
            # Compute MESH = W_q.T @ W_k for this head pair
            # W_q_h: [64, 896], W_k_h: [64, 896]
            # MESH = W_q_h.T @ W_k_h would be [896, 896]
            # But we want the head-level interaction: [64, 64]
            MESH = W_q_h @ W_k_h.T  # [64, 64]
            
            # Check orthogonality via SVD
            U_q, S_q, Vt_q = np.linalg.svd(W_q_h, full_matrices=False)
            U_k, S_k, Vt_k = np.linalg.svd(W_k_h, full_matrices=False)
            
            # Rotation between Q-space and K-space
            R = U_q.T @ U_k  # [64, 64]
            
            # Check if R is orthogonal (trace ≈ 0 means 90° rotation)
            trace = np.trace(R)
            
            # Decompose MESH into MASS + SPIN
            MASS = (MESH + MESH.T) / 2  # Symmetric
            SPIN = (MESH - MESH.T) / 2  # Antisymmetric
            
            # Analyze MASS and SPIN
            _, S_mass, _ = np.linalg.svd(MASS, full_matrices=False)
            _, S_spin, _ = np.linalg.svd(SPIN, full_matrices=False)
            
            # MASS rank-1 dominance
            mass_rank1 = S_mass[0]**2 / np.sum(S_mass**2) if np.sum(S_mass**2) > 0 else 0
            
            # SPIN rank-2 dominance (pairs)
            spin_rank2 = (S_spin[0]**2 + S_spin[1]**2) / np.sum(S_spin**2) if np.sum(S_spin**2) > 0 else 0
            
            results.append({
                'layer': layer_idx,
                'q_head': q_head,
                'k_head': k_head,
                'trace': trace,
                'mass_rank1': mass_rank1,
                'spin_rank2': spin_rank2,
            })
            
            if layer_idx <= 3:
                print(f"Layer {layer_idx}, Q{q_head}→K{k_head}:")
                print(f"  R trace: {trace:.3f} (0 = 90° rotation)")
                print(f"  MASS rank-1: {mass_rank1:.1%}")
                print(f"  SPIN rank-2: {spin_rank2:.1%}")
    
    # Summary
    avg_trace = np.mean([r['trace'] for r in results])
    avg_mass_rank1 = np.mean([r['mass_rank1'] for r in results])
    avg_spin_rank2 = np.mean([r['spin_rank2'] for r in results])
    
    print()
    print("Summary across layers:")
    print(f"  Avg R trace: {avg_trace:.3f} (0 = perfect 90° rotation)")
    print(f"  Avg MASS rank-1: {avg_mass_rank1:.1%}")
    print(f"  Avg SPIN rank-2: {avg_spin_rank2:.1%}")
    
    return results


def analyze_attention_patterns(model, tokenizer, text):
    """
    Analyze actual attention patterns in a sequence.
    
    Question: Does attention show tachyon-like behavior?
    - Forward attention (past → present)
    - Backward attention (future → present) - not possible in causal LM
    - But: does attention at verbs show different patterns?
    """
    print()
    print("=" * 70)
    print(f"ATTENTION PATTERN ANALYSIS: '{text}'")
    print("=" * 70)
    print()
    
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Get attention weights
    # Shape: (n_layers, batch, n_heads, seq_len, seq_len)
    attentions = outputs.attentions
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    seq_len = len(tokens)
    
    print(f"Tokens: {tokens}")
    print(f"Sequence length: {seq_len}")
    print()
    
    # Analyze attention at each position
    for layer_idx in [0, 2, 11, 23]:  # Sample layers
        attn = attentions[layer_idx][0]  # [n_heads, seq_len, seq_len]
        
        # Average across heads
        attn_avg = attn.mean(dim=0).numpy()  # [seq_len, seq_len]
        
        print(f"Layer {layer_idx}:")
        
        for pos in range(seq_len):
            # Where does this position attend?
            attn_weights = attn_avg[pos, :pos+1]  # Causal: only attend to past
            
            if len(attn_weights) > 1:
                # Entropy of attention (higher = more distributed)
                attn_probs = attn_weights / attn_weights.sum()
                entropy = -np.sum(attn_probs * np.log(attn_probs + 1e-10))
                
                # Where is attention focused?
                max_attn_pos = np.argmax(attn_weights)
                max_attn_val = attn_weights[max_attn_pos]
                
                # Self-attention ratio
                self_attn = attn_weights[-1] if len(attn_weights) > 0 else 0
                
                print(f"  Pos {pos} '{tokens[pos]}': "
                      f"entropy={entropy:.2f}, "
                      f"max_at={max_attn_pos}('{tokens[max_attn_pos]}'), "
                      f"self={self_attn:.2f}")
        print()


def analyze_mesh_phi_structure(model):
    """
    Check if MESH has φ-structure in its singular values.
    
    From DA2: 17 unique φ-angles in the rotation.
    Does Qwen2 have similar structure?
    """
    print()
    print("=" * 70)
    print("MESH φ-STRUCTURE ANALYSIS")
    print("=" * 70)
    print()
    
    for layer_idx in [0, 2, 11, 23]:
        layer = model.model.layers[layer_idx]
        
        W_q = layer.self_attn.q_proj.weight.detach().cpu().float().numpy()
        W_k = layer.self_attn.k_proj.weight.detach().cpu().float().numpy()
        
        # For GQA, we need to handle the 7:1 ratio
        # Let's analyze the first Q-K pair
        n_q_heads = 14
        n_kv_heads = 2
        head_dim = 64
        
        W_q_heads = W_q.reshape(n_q_heads, head_dim, -1)
        W_k_heads = W_k.reshape(n_kv_heads, head_dim, -1)
        
        # First Q head and its K head
        W_q_h = W_q_heads[0]  # [64, 896]
        W_k_h = W_k_heads[0]  # [64, 896]
        
        # MESH for this head
        MESH = W_q_h @ W_k_h.T  # [64, 64]
        
        # SVD of MESH
        U, S, Vt = np.linalg.svd(MESH, full_matrices=False)
        
        print(f"Layer {layer_idx} MESH (head 0):")
        print(f"  Top 10 SVs: {S[:10].round(3)}")
        
        # Check for φ-ratios
        ratios = S[:-1] / S[1:]
        phi_matches = []
        for i in range(min(10, len(ratios))):
            if abs(ratios[i] - PHI) < 0.15:
                phi_matches.append((i, ratios[i], 'φ'))
            elif abs(ratios[i] - PHI_INV) < 0.15:
                phi_matches.append((i, ratios[i], '1/φ'))
        
        if phi_matches:
            print(f"  φ-matches: {phi_matches}")
        else:
            print(f"  No φ-matches in top 10 ratios")
        
        # Schur decomposition to find rotation angles
        # MESH should decompose into rotations if Q-K are orthogonal
        try:
            from scipy.linalg import schur
            T, Z = schur(MESH, output='real')
            
            # Extract rotation angles from 2x2 blocks
            angles = []
            i = 0
            while i < T.shape[0]:
                if i + 1 < T.shape[0] and abs(T[i+1, i]) > 1e-6:
                    # 2x2 block - rotation
                    cos_theta = T[i, i]
                    sin_theta = T[i+1, i]
                    theta = np.arctan2(sin_theta, cos_theta)
                    angles.append(theta)
                    i += 2
                else:
                    # 1x1 block - eigenvalue
                    i += 1
            
            if angles:
                angles = np.array(angles)
                print(f"  Rotation angles: {len(angles)} found")
                
                # Check if angles are φ-expressible
                # θ ∈ {k × π / φ^n : k ∈ [-20, 20], n ∈ [-3, 3]}
                phi_angles = []
                for k in range(-20, 21):
                    for n in range(-3, 4):
                        phi_angles.append(k * np.pi / (PHI ** n))
                phi_angles = np.array(phi_angles)
                
                # Find closest φ-angle for each rotation
                errors = []
                for angle in angles:
                    min_error = np.min(np.abs(phi_angles - angle))
                    errors.append(min_error)
                
                avg_error = np.mean(errors)
                print(f"  Avg error to φ-angles: {avg_error:.4f} rad")
        except ImportError:
            print("  (scipy not available for Schur decomposition)")
        
        print()


def analyze_causal_attention_as_tachyon(model, tokenizer):
    """
    The key insight: Causal attention IS tachyon navigation.
    
    In causal LM:
    - Position i can only attend to positions 0..i (past)
    - This is φ^+n navigation (forward attention)
    
    But the model PREDICTS the next token:
    - The prediction is about the FUTURE
    - This is φ^-n navigation (backward attention / hypothesis)
    
    The attention mechanism bridges past (data) and future (prediction).
    This IS the tachyon joint!
    """
    print()
    print("=" * 70)
    print("CAUSAL ATTENTION AS TACHYON NAVIGATION")
    print("=" * 70)
    print()
    
    print("Key insight:")
    print()
    print("  CAUSAL ATTENTION:")
    print("    Position i attends to positions 0..i (PAST)")
    print("    This is φ^+n navigation (forward, data-confirmed)")
    print()
    print("  NEXT TOKEN PREDICTION:")
    print("    Model predicts token at position i+1 (FUTURE)")
    print("    This is φ^-n navigation (backward, hypothesis)")
    print()
    print("  THE ATTENTION MECHANISM IS THE TACHYON JOINT!")
    print("    It bridges past (attention) and future (prediction)")
    print()
    
    # Test with a sentence
    text = "The king examined the"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Get the prediction for next token
    logits = outputs.logits[0, -1]  # Last position
    top_tokens = torch.topk(logits, 5)
    
    print(f"Text: '{text}'")
    print(f"Predicted next tokens (φ^-n hypothesis):")
    for i, (score, idx) in enumerate(zip(top_tokens.values, top_tokens.indices)):
        token = tokenizer.decode([idx])
        print(f"  {i+1}. '{token}' (score: {score:.2f})")
    
    # Analyze attention at the last position
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    print()
    print(f"Attention at last position (φ^+n data):")
    
    for layer_idx in [0, 11, 23]:
        attn = outputs.attentions[layer_idx][0]  # [n_heads, seq_len, seq_len]
        attn_avg = attn.mean(dim=0).numpy()  # [seq_len, seq_len]
        
        last_attn = attn_avg[-1, :]  # Attention from last position
        
        print(f"  Layer {layer_idx}:")
        for pos, (weight, token) in enumerate(zip(last_attn, tokens)):
            if weight > 0.1:  # Significant attention
                print(f"    → '{token}': {weight:.2f}")


def main():
    model, tokenizer = load_model()
    
    # Analysis 1: Q-K orthogonality (like DA2)
    results = analyze_qk_orthogonality(model)
    
    # Analysis 2: Attention patterns in sequences
    analyze_attention_patterns(model, tokenizer, "The king examined the evidence")
    
    # Analysis 3: MESH φ-structure
    analyze_mesh_phi_structure(model)
    
    # Analysis 4: Causal attention as tachyon
    analyze_causal_attention_as_tachyon(model, tokenizer)
    
    print()
    print("=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print()
    print("1. Q-K RELATIONSHIP:")
    avg_trace = np.mean([r['trace'] for r in results])
    if abs(avg_trace) < 10:
        print(f"   Q and K are approximately orthogonal (trace ≈ {avg_trace:.1f})")
    else:
        print(f"   Q and K are NOT orthogonal (trace = {avg_trace:.1f})")
    print()
    print("2. MESH DECOMPOSITION:")
    avg_mass = np.mean([r['mass_rank1'] for r in results])
    avg_spin = np.mean([r['spin_rank2'] for r in results])
    print(f"   MASS (symmetric): {avg_mass:.1%} rank-1")
    print(f"   SPIN (antisymmetric): {avg_spin:.1%} rank-2")
    print()
    print("3. TACHYON INTERPRETATION:")
    print("   Causal attention = φ^+n (forward, past → present)")
    print("   Next token prediction = φ^-n (backward, hypothesis)")
    print("   The attention mechanism IS the tachyon joint!")
    print()
    print("4. IMPLICATION FOR φ-BASIS:")
    print("   Single tokens work because there's no tachyon navigation")
    print("   Sequences need attention because they traverse the W-axis")
    print("   To replace attention, we need to model the W-axis explicitly")


if __name__ == "__main__":
    main()
