#!/usr/bin/env python3
"""
Transformation Space Probing with 12D Clock
=============================================

Key insight shift:
- Hidden states are numerous (one per token sequence)
- But TRANSFORMATIONS are constrained (tetromino: ~300 patterns)

The transformer applies: hidden_out = f(hidden_in, weights)

If weights are on φ-lattice with finite patterns, then f() is constrained.
We can probe the transformation space using the 12D clock.

The 12D clock uses ratios like φ, √2, π, e to create quasi-periodic
patterns that systematically cover a space without repetition.

Strategy:
1. Use clock phases to generate "probe vectors" in hidden space
2. Apply transformer layers to probe vectors
3. Analyze the transformation (output - input)
4. Cluster transformations to find the finite vocabulary

If we can enumerate all transformations, we can precompute the model!

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)

# 12D clock ratios (from ribbon_attention.py)
CLOCK_RATIOS_12D = {
    'phi': PHI,
    'phi_sq': PHI ** 2,
    'sqrt2': np.sqrt(2),
    'sqrt3': np.sqrt(3),
    'sqrt5': np.sqrt(5),
    'pi': np.pi,
    'e': np.e,
    'ln2': np.log(2),
    'phi_inv': 1 / PHI,
    'sqrt2_inv': 1 / np.sqrt(2),
    'pi_inv': 1 / np.pi,
    'e_inv': 1 / np.e,
}


def get_clock_vector(n: int) -> np.ndarray:
    """Get 12D clock phase vector at position n."""
    return np.array([
        (n * ratio) % 1.0
        for ratio in CLOCK_RATIOS_12D.values()
    ])


def clock_to_hidden_probe(clock_vec: np.ndarray, hidden_dim: int, scale: float = 10.0) -> np.ndarray:
    """
    Convert 12D clock vector to a probe vector in hidden space.
    
    Uses the clock phases to generate a structured probe that
    systematically covers the hidden space.
    """
    # Expand 12D to hidden_dim using harmonic expansion
    probe = np.zeros(hidden_dim)
    
    for i in range(hidden_dim):
        # Each dimension gets a unique combination of clock phases
        dim_phase = 0
        for j, phase in enumerate(clock_vec):
            # Use different harmonics for each dimension
            harmonic = (i * (j + 1)) % hidden_dim
            dim_phase += phase * np.sin(2 * np.pi * harmonic / hidden_dim)
        
        probe[i] = scale * np.tanh(dim_phase)
    
    return probe


class TransformationProber:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.hidden_dim = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
    
    def _apply_single_layer(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply a single transformer layer."""
        layer = self.model.model.layers[layer_idx]
        
        # Create minimal inputs with proper position embeddings
        device = hidden.device
        position_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        
        # Get rotary embeddings
        rotary_emb = self.model.model.rotary_emb
        cos, sin = rotary_emb(hidden.unsqueeze(0).unsqueeze(0), position_ids)
        position_embeddings = (cos, sin)
        
        with torch.no_grad():
            # Apply layer
            output = layer(
                hidden.unsqueeze(0).unsqueeze(0),  # Add batch and seq dims
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            return output[0].squeeze()
    
    def probe_layer_transformation(self, layer_idx: int, n_probes: int = 100) -> Dict:
        """
        Probe a single layer's transformation using clock-generated inputs.
        
        Returns analysis of the transformation space.
        """
        print(f"\n--- Probing Layer {layer_idx} with {n_probes} clock positions ---")
        
        device = next(self.model.parameters()).device
        
        transformations = []
        input_norms = []
        output_norms = []
        
        for n in range(n_probes):
            # Generate probe from clock
            clock_vec = get_clock_vector(n * 137)  # Use prime spacing
            probe = clock_to_hidden_probe(clock_vec, self.hidden_dim)
            
            # Convert to tensor
            probe_tensor = torch.tensor(probe, dtype=torch.float16, device=device)
            
            # Apply layer
            output = self._apply_single_layer(probe_tensor, layer_idx)
            
            # Compute transformation
            output_np = output.float().cpu().numpy()
            transformation = output_np - probe
            
            transformations.append(transformation)
            input_norms.append(np.linalg.norm(probe))
            output_norms.append(np.linalg.norm(output_np))
        
        transformations = np.array(transformations)
        
        # Analyze transformation space
        # 1. SVD to find principal transformation directions
        U, S, Vt = np.linalg.svd(transformations, full_matrices=False)
        
        # Variance explained
        var_explained = S**2 / (S**2).sum()
        cumvar = np.cumsum(var_explained)
        
        # 2. How many dimensions needed for 90%, 99%?
        dims_90 = np.searchsorted(cumvar, 0.90) + 1
        dims_99 = np.searchsorted(cumvar, 0.99) + 1
        
        print(f"  Transformation analysis:")
        print(f"    Mean input norm: {np.mean(input_norms):.2f}")
        print(f"    Mean output norm: {np.mean(output_norms):.2f}")
        print(f"    Mean transformation norm: {np.mean(np.linalg.norm(transformations, axis=1)):.2f}")
        print(f"    Dimensions for 90% variance: {dims_90}")
        print(f"    Dimensions for 99% variance: {dims_99}")
        print(f"    Top 5 singular values: {S[:5].round(2)}")
        
        return {
            'layer': layer_idx,
            'transformations': transformations,
            'S': S,
            'Vt': Vt,
            'dims_90': dims_90,
            'dims_99': dims_99,
            'cumvar': cumvar,
        }
    
    def probe_full_model(self, n_probes: int = 50) -> Dict:
        """
        Probe the full model transformation (all layers).
        """
        print(f"\n--- Probing Full Model with {n_probes} clock positions ---")
        
        device = next(self.model.parameters()).device
        
        transformations = []
        
        for n in range(n_probes):
            if n % 10 == 0:
                print(f"  Probe {n}/{n_probes}...")
            
            # Generate probe from clock
            clock_vec = get_clock_vector(n * 137)
            probe = clock_to_hidden_probe(clock_vec, self.hidden_dim)
            
            # Apply all layers
            hidden = torch.tensor(probe, dtype=torch.float16, device=device)
            
            for layer_idx in range(self.n_layers):
                hidden = self._apply_single_layer(hidden, layer_idx)
            
            # Total transformation
            output_np = hidden.float().cpu().numpy()
            transformation = output_np - probe
            transformations.append(transformation)
        
        transformations = np.array(transformations)
        
        # SVD analysis
        U, S, Vt = np.linalg.svd(transformations, full_matrices=False)
        var_explained = S**2 / (S**2).sum()
        cumvar = np.cumsum(var_explained)
        
        dims_90 = np.searchsorted(cumvar, 0.90) + 1
        dims_99 = np.searchsorted(cumvar, 0.99) + 1
        
        print(f"\n  Full model transformation analysis:")
        print(f"    Dimensions for 90% variance: {dims_90}")
        print(f"    Dimensions for 99% variance: {dims_99}")
        print(f"    Top 10 singular values: {S[:10].round(2)}")
        
        return {
            'transformations': transformations,
            'S': S,
            'Vt': Vt,
            'dims_90': dims_90,
            'dims_99': dims_99,
        }
    
    def compare_token_vs_clock_transformations(self, n_samples: int = 50) -> Dict:
        """
        Compare transformations from real tokens vs clock probes.
        
        Key question: Do clock probes cover the same transformation space
        as real token inputs?
        """
        print(f"\n--- Comparing Token vs Clock Transformations ---")
        
        device = next(self.model.parameters()).device
        rotary_emb = self.model.model.rotary_emb
        
        # Get transformations from real tokens
        token_transforms = []
        for i in range(n_samples):
            token_id = np.random.randint(0, self.tokenizer.vocab_size)
            
            # Get embedding (keep as float16)
            embedding = self.model.model.embed_tokens.weight[token_id]
            embedding_np = embedding.detach().float().cpu().numpy()
            
            # Apply all layers
            hidden = embedding.unsqueeze(0).unsqueeze(0)
            position_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
            cos, sin = rotary_emb(hidden, position_ids)
            position_embeddings = (cos, sin)
            
            with torch.no_grad():
                for layer in self.model.model.layers:
                    output = layer(hidden, position_ids=position_ids, position_embeddings=position_embeddings)
                    hidden = output[0]
            
            # Transformation
            transform = hidden.squeeze().float().cpu().numpy() - embedding_np
            token_transforms.append(transform)
        
        token_transforms = np.array(token_transforms)
        
        # Get transformations from clock probes
        clock_transforms = []
        for n in range(n_samples):
            clock_vec = get_clock_vector(n * 137)
            probe = clock_to_hidden_probe(clock_vec, self.hidden_dim)
            
            hidden = torch.tensor(probe, dtype=torch.float16, device=device)
            hidden = hidden.unsqueeze(0).unsqueeze(0)
            position_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
            cos, sin = rotary_emb(hidden, position_ids)
            position_embeddings = (cos, sin)
            
            with torch.no_grad():
                for layer in self.model.model.layers:
                    output = layer(hidden, position_ids=position_ids, position_embeddings=position_embeddings)
                    hidden = output[0]
            
            transform = hidden.squeeze().float().cpu().numpy() - probe
            clock_transforms.append(transform)
        
        clock_transforms = np.array(clock_transforms)
        
        # Compare the spaces
        # 1. SVD of each
        _, S_token, Vt_token = np.linalg.svd(token_transforms, full_matrices=False)
        _, S_clock, Vt_clock = np.linalg.svd(clock_transforms, full_matrices=False)
        
        # 2. How much do the principal directions overlap?
        # Compute alignment between top-k directions
        k = 20
        alignment = np.abs(Vt_token[:k] @ Vt_clock[:k].T)
        mean_alignment = alignment.max(axis=1).mean()
        
        print(f"\n  Token transformations:")
        print(f"    Top 5 singular values: {S_token[:5].round(2)}")
        
        print(f"\n  Clock transformations:")
        print(f"    Top 5 singular values: {S_clock[:5].round(2)}")
        
        print(f"\n  Alignment (top {k} directions): {mean_alignment:.3f}")
        print(f"    (1.0 = perfect alignment, 0.0 = orthogonal)")
        
        # 3. Can clock basis reconstruct token transformations?
        print(f"\n  Reconstruction test:")
        
        for k in [10, 20, 30, 40]:
            if k > len(S_clock):
                continue
            
            # Project token transforms onto clock basis
            projected = token_transforms @ Vt_clock[:k].T @ Vt_clock[:k]
            
            # Reconstruction error
            errors = np.linalg.norm(token_transforms - projected, axis=1)
            orig_norms = np.linalg.norm(token_transforms, axis=1)
            rel_error = (errors / (orig_norms + 1e-10)).mean()
            
            print(f"    k={k}: {(1-rel_error)*100:.1f}% reconstruction")
        
        return {
            'token_transforms': token_transforms,
            'clock_transforms': clock_transforms,
            'S_token': S_token,
            'S_clock': S_clock,
            'Vt_token': Vt_token,
            'Vt_clock': Vt_clock,
            'alignment': mean_alignment,
        }


def main():
    print("=" * 70)
    print("TRANSFORMATION SPACE PROBING WITH 12D CLOCK")
    print("=" * 70)
    print("""
Key insight: Probe the TRANSFORMATION space, not hidden state space.

The transformer applies constrained transformations (tetromino weights).
The 12D clock provides systematic coverage of the input space.

If clock probes cover the same transformation space as real tokens,
we can use the clock basis to precompute all transformations!
""")
    
    prober = TransformationProber()
    
    # 1. Probe a single layer
    layer_results = prober.probe_layer_transformation(layer_idx=14, n_probes=100)
    
    # 2. Compare token vs clock transformations
    comparison = prober.compare_token_vs_clock_transformations(n_samples=50)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"""
Layer 14 transformation:
  - {layer_results['dims_90']} dimensions capture 90% of variance
  - {layer_results['dims_99']} dimensions capture 99% of variance

Token vs Clock comparison:
  - Alignment: {comparison['alignment']:.3f}
  - Clock probes {'DO' if comparison['alignment'] > 0.5 else 'DO NOT'} cover token transformation space

IMPLICATION:
""")
    
    if comparison['alignment'] > 0.5:
        print("  The clock basis CAN represent token transformations!")
        print("  We can precompute transformations using clock probes.")
    else:
        print("  Clock probes don't fully cover token space.")
        print("  Need different probing strategy.")


if __name__ == "__main__":
    main()
