#!/usr/bin/env python3
"""
Test if Transformer Hidden States are φ-Coordinates
=====================================================

If the transformer is computing in φ-space, then:
1. Hidden state values should cluster at φ-levels
2. The distribution should show peaks at φ^n
3. Gate values should be near φ-boundaries (±ln(φ))
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from collections import Counter

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)


def analyze_phi_structure(values: np.ndarray, name: str):
    """Analyze if values cluster at φ-levels."""
    values = values.flatten()
    values = values[np.abs(values) > 1e-10]  # Remove near-zero
    
    # Compute log_φ of absolute values
    log_phi = np.log(np.abs(values)) / LN_PHI
    
    # Distance to nearest integer (φ-level)
    residuals = np.abs(log_phi - np.round(log_phi))
    
    # What fraction are "on" a φ-level (residual < 0.1)?
    on_level = np.mean(residuals < 0.1)
    
    # Distribution of levels
    levels = np.round(log_phi).astype(int)
    level_counts = Counter(levels)
    top_levels = level_counts.most_common(5)
    
    print(f"\n  {name}:")
    print(f"    Values analyzed: {len(values):,}")
    print(f"    On φ-level (residual < 0.1): {on_level*100:.1f}%")
    print(f"    Mean residual: {np.mean(residuals):.4f}")
    print(f"    Top levels: {top_levels}")
    
    return on_level, np.mean(residuals), levels


def main():
    print("=" * 70)
    print("TESTING φ-STRUCTURE IN TRANSFORMER HIDDEN STATES")
    print("=" * 70)
    
    config = AutoConfig.from_pretrained("Qwen/Qwen2-7B-Instruct")
    config._attn_implementation = "eager"
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    device = next(model.parameters()).device
    
    # Test text
    text = "The quick brown fox jumps over the lazy dog."
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    
    print(f"\nTest text: '{text}'")
    print(f"Tokens: {ids.shape[1]}")
    
    # Capture hidden states and gate values
    captured = {}
    
    def capture_gate(layer_idx):
        def hook(module, input, output):
            # Gate projection output
            captured[f'gate_{layer_idx}'] = input[0].detach().float().cpu().numpy()
        return hook
    
    hooks = []
    for layer_idx in [0, 7, 14, 21, 27]:
        h = model.model.layers[layer_idx].mlp.gate_proj.register_forward_hook(capture_gate(layer_idx))
        hooks.append(h)
    
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    
    for h in hooks:
        h.remove()
    
    # Analyze hidden states
    print("\n" + "=" * 70)
    print("1. HIDDEN STATE φ-STRUCTURE")
    print("=" * 70)
    
    for layer_idx in [0, 7, 14, 21, 27]:
        hidden = out.hidden_states[layer_idx + 1][0].float().cpu().numpy()
        analyze_phi_structure(hidden, f"Layer {layer_idx} hidden states")
    
    # Analyze gate inputs (what goes into SiLU)
    print("\n" + "=" * 70)
    print("2. GATE INPUT φ-STRUCTURE (before SiLU)")
    print("=" * 70)
    
    for layer_idx in [0, 7, 14, 21, 27]:
        gate_input = captured[f'gate_{layer_idx}']
        analyze_phi_structure(gate_input, f"Layer {layer_idx} gate input")
    
    # Check if gate values are near φ-boundaries
    print("\n" + "=" * 70)
    print("3. GATE VALUES vs φ-BOUNDARIES")
    print("=" * 70)
    
    print(f"\n  φ-boundaries: ±ln(φ) = ±{LN_PHI:.4f}")
    print(f"  These define the linear regime of sigmoid/SiLU")
    
    for layer_idx in [0, 7, 14, 21, 27]:
        gate_input = captured[f'gate_{layer_idx}'].flatten()
        
        # What fraction is in the linear regime?
        in_linear = np.mean(np.abs(gate_input) < LN_PHI)
        
        # What fraction is near the boundaries?
        near_boundary = np.mean(np.abs(np.abs(gate_input) - LN_PHI) < 0.1)
        
        print(f"\n  Layer {layer_idx}:")
        print(f"    In linear regime (|x| < ln(φ)): {in_linear*100:.1f}%")
        print(f"    Near boundary (||x| - ln(φ)| < 0.1): {near_boundary*100:.1f}%")
        print(f"    Gate std: {np.std(gate_input):.4f}")
        print(f"    Gate range: [{np.min(gate_input):.4f}, {np.max(gate_input):.4f}]")
    
    # Analyze embeddings
    print("\n" + "=" * 70)
    print("4. EMBEDDING φ-STRUCTURE")
    print("=" * 70)
    
    embeddings = model.model.embed_tokens.weight.data.float().cpu().numpy()
    analyze_phi_structure(embeddings, "Embeddings")
    
    # Check embedding norms
    norms = np.linalg.norm(embeddings, axis=1)
    log_phi_norms = np.log(norms + 1e-10) / LN_PHI
    norm_levels = np.round(log_phi_norms).astype(int)
    
    print(f"\n  Embedding norm analysis:")
    print(f"    Mean norm: {np.mean(norms):.4f}")
    print(f"    Mean log_φ(norm): {np.mean(log_phi_norms):.4f}")
    print(f"    Most common norm levels: {Counter(norm_levels).most_common(5)}")
    
    # Final analysis
    print("\n" + "=" * 70)
    print("5. SUMMARY")
    print("=" * 70)
    
    print("""
  FINDINGS:
  
  If values cluster at φ-levels (residual < 0.1 for >50%), 
  this supports the φ-computer hypothesis.
  
  If gate values cluster near ±ln(φ) boundaries,
  this suggests the model operates at the φ-transition points.
  
  The key question: Is the ~20% φ-alignment we see in weights
  also present in activations, or do activations show MORE alignment?
""")


if __name__ == "__main__":
    main()
