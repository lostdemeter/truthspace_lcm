"""
Analyze whether model intermediate outputs cluster at φ-levels.

Key question: Do the OUTPUTS of transformer layers also live on the φ-lattice,
or is the lattice structure only in the weights?
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)


def analyze_phi_clustering(values: np.ndarray, name: str):
    """Check if values cluster at φ^k levels."""
    values = values.flatten()
    values_abs = np.abs(values)
    values_abs = values_abs[values_abs > 1e-10]
    
    if len(values_abs) == 0:
        print(f"{name}: all zeros")
        return
    
    # Compute actual levels
    levels_actual = np.log(values_abs) / LOG_PHI
    level_fractions = levels_actual - np.round(levels_actual)
    
    # Std of fractions: 0 = perfect lattice, 0.29 = uniform random
    fraction_std = np.std(level_fractions)
    
    # Distribution of levels
    levels_rounded = np.round(levels_actual).astype(int)
    unique, counts = np.unique(levels_rounded, return_counts=True)
    
    print(f"\n{name}:")
    print(f"  Value range: [{values.min():.4f}, {values.max():.4f}]")
    print(f"  Level fraction std: {fraction_std:.3f} (0=lattice, 0.29=random)")
    print(f"  Lattice fit: {'GOOD' if fraction_std < 0.15 else 'MODERATE' if fraction_std < 0.25 else 'POOR'}")
    
    # Show top levels
    top_idx = np.argsort(counts)[::-1][:5]
    print(f"  Top levels: ", end="")
    for i in top_idx:
        print(f"φ^{unique[i]}({counts[i]/len(levels_actual)*100:.1f}%) ", end="")
    print()


def main():
    print("Loading model...")
    model_name = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    
    # Test prompt
    prompt = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print(f"\nPrompt: {prompt}")
    print(f"Tokens: {inputs.input_ids.shape[1]}")
    
    # Analyze weights (should cluster at φ^-9)
    print("\n" + "=" * 60)
    print("WEIGHTS (should cluster at φ^-9)")
    print("=" * 60)
    
    layer = model.model.layers[0]
    analyze_phi_clustering(layer.mlp.gate_proj.weight.data.numpy(), "Layer 0 MLP gate")
    analyze_phi_clustering(layer.self_attn.q_proj.weight.data.numpy(), "Layer 0 Q proj")
    
    # Run forward pass and capture intermediate outputs
    print("\n" + "=" * 60)
    print("INTERMEDIATE OUTPUTS (do they cluster?)")
    print("=" * 60)
    
    with torch.no_grad():
        # Get embeddings
        hidden = model.model.embed_tokens(inputs.input_ids)
        analyze_phi_clustering(hidden.numpy(), "Embeddings")
        
        # Run through first few layers
        for layer_idx in range(3):
            layer = model.model.layers[layer_idx]
            
            # Pre-norm
            normed = layer.input_layernorm(hidden)
            analyze_phi_clustering(normed.numpy(), f"Layer {layer_idx} post-norm")
            
            # Attention
            attn_out, _, _ = layer.self_attn(
                normed,
                attention_mask=None,
                position_ids=None,
            )
            analyze_phi_clustering(attn_out.numpy(), f"Layer {layer_idx} attn output")
            
            # Residual
            hidden = hidden + attn_out
            
            # MLP
            normed2 = layer.post_attention_layernorm(hidden)
            
            # Gate and up
            gate = layer.mlp.gate_proj(normed2)
            up = layer.mlp.up_proj(normed2)
            analyze_phi_clustering(gate.numpy(), f"Layer {layer_idx} MLP gate")
            analyze_phi_clustering(up.numpy(), f"Layer {layer_idx} MLP up")
            
            # SiLU(gate) * up
            activated = torch.nn.functional.silu(gate) * up
            analyze_phi_clustering(activated.numpy(), f"Layer {layer_idx} MLP activated")
            
            # Down projection
            mlp_out = layer.mlp.down_proj(activated)
            analyze_phi_clustering(mlp_out.numpy(), f"Layer {layer_idx} MLP output")
            
            # Residual
            hidden = hidden + mlp_out
            analyze_phi_clustering(hidden.numpy(), f"Layer {layer_idx} final hidden")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
If weights cluster (std < 0.15) but outputs don't (std > 0.25):
  → The φ-lattice is a property of WEIGHTS, not activations
  → We can't expect intermediate values to snap to lattice
  → Need different approach for geometric navigation

If both cluster:
  → The geometric structure propagates through the network
  → Lattice-based navigation is viable
""")


if __name__ == "__main__":
    main()
