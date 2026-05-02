#!/usr/bin/env python3
"""
Layer 27 Analysis: Why does rank-1 fail here?
==============================================

Layer 27 has 90% error with rank-1 approximation.
This is the final layer before the LM head.

Questions:
1. What's the true rank of layer 27's transformation?
2. Is there a different structure we can exploit?
3. Can we cache layer 27 outputs directly?

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


class Layer27Analyzer:
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
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
    
    def get_layer_outputs(self, token_ids):
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return [h[0, -1, :].float().cpu().numpy() for h in outputs.hidden_states]
    
    def analyze_layer27_rank(self, n_samples: int = 200):
        """Analyze the true rank of layer 27's transformation."""
        print(f"\n--- Layer 27 Rank Analysis ({n_samples} samples) ---")
        
        transforms = []
        
        for i in range(n_samples):
            if i % 50 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            token_id = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                outputs = self.get_layer_outputs([token_id])
                transform = outputs[-1] - outputs[-2]  # Layer 27 = final - second-to-last
                transforms.append(transform)
            except:
                continue
        
        transforms = np.array(transforms)
        
        # SVD
        U, S, Vt = np.linalg.svd(transforms, full_matrices=False)
        
        # Variance explained
        var_explained = S**2 / (S**2).sum()
        cumvar = np.cumsum(var_explained)
        
        print(f"\n  Singular values (top 20): {S[:20].round(1)}")
        print(f"\n  Variance explained:")
        for k in [1, 2, 5, 10, 20, 50, 100]:
            if k <= len(cumvar):
                print(f"    k={k}: {cumvar[k-1]*100:.1f}%")
        
        # How many dimensions for 90%, 99%?
        dims_90 = np.searchsorted(cumvar, 0.90) + 1
        dims_99 = np.searchsorted(cumvar, 0.99) + 1
        
        print(f"\n  Dimensions for 90% variance: {dims_90}")
        print(f"  Dimensions for 99% variance: {dims_99}")
        
        return {
            'S': S,
            'Vt': Vt,
            'dims_90': dims_90,
            'dims_99': dims_99,
        }
    
    def analyze_layer27_structure(self, n_samples: int = 100):
        """
        What IS layer 27 doing?
        
        Hypothesis: It's projecting onto the vocabulary space.
        """
        print(f"\n--- Layer 27 Structure Analysis ---")
        
        # Collect inputs and outputs
        inputs = []
        outputs = []
        
        for i in range(n_samples):
            token_id = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                layer_outputs = self.get_layer_outputs([token_id])
                inputs.append(layer_outputs[-2])  # Input to layer 27
                outputs.append(layer_outputs[-1])  # Output of layer 27
            except:
                continue
        
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        
        # What's the relationship between input and output?
        print(f"\n  Input/Output statistics:")
        print(f"    Input mean norm: {np.linalg.norm(inputs, axis=1).mean():.1f}")
        print(f"    Output mean norm: {np.linalg.norm(outputs, axis=1).mean():.1f}")
        
        # Correlation between input and output
        correlations = []
        for i in range(len(inputs)):
            corr = np.corrcoef(inputs[i], outputs[i])[0, 1]
            correlations.append(corr)
        
        print(f"    Mean input-output correlation: {np.mean(correlations):.3f}")
        
        # Is the output mostly the input?
        residuals = outputs - inputs
        residual_norms = np.linalg.norm(residuals, axis=1)
        output_norms = np.linalg.norm(outputs, axis=1)
        
        print(f"    Mean residual/output ratio: {(residual_norms / output_norms).mean():.3f}")
        
        # What's the structure of the residual?
        print(f"\n  Residual (output - input) analysis:")
        
        _, S_res, Vt_res = np.linalg.svd(residuals, full_matrices=False)
        var_explained = S_res**2 / (S_res**2).sum()
        cumvar = np.cumsum(var_explained)
        
        print(f"    Top 5 singular values: {S_res[:5].round(1)}")
        print(f"    Dims for 90% variance: {np.searchsorted(cumvar, 0.90) + 1}")
        print(f"    Dims for 99% variance: {np.searchsorted(cumvar, 0.99) + 1}")
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            'residuals': residuals,
        }
    
    def test_direct_caching(self, n_samples: int = 100):
        """
        What if we just cache the FINAL hidden state directly?
        
        This is what we did before with 512x compression.
        """
        print(f"\n--- Direct Caching Test ---")
        
        # Collect final hidden states
        final_hiddens = []
        token_ids_list = []
        
        for i in range(n_samples):
            token_id = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                outputs = self.get_layer_outputs([token_id])
                final_hiddens.append(outputs[-1])
                token_ids_list.append(token_id)
            except:
                continue
        
        final_hiddens = np.array(final_hiddens)
        
        # SVD for compression
        mean_hidden = final_hiddens.mean(axis=0)
        centered = final_hiddens - mean_hidden
        
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # Test reconstruction at different k
        print(f"\n  Basis compression test:")
        
        for k in [10, 20, 50, 100]:
            if k > len(S):
                continue
            
            correct = 0
            for i in range(len(final_hiddens)):
                # Project onto basis
                coeffs = centered[i] @ Vt[:k].T
                reconstructed = mean_hidden + coeffs @ Vt[:k]
                
                # Decode
                true_logits = np.dot(self.lm_head, final_hiddens[i])
                recon_logits = np.dot(self.lm_head, reconstructed)
                
                if np.argmax(true_logits) == np.argmax(recon_logits):
                    correct += 1
            
            accuracy = correct / len(final_hiddens)
            storage = k * 4  # bytes per token
            print(f"    k={k}: {accuracy*100:.1f}% accuracy, {storage} bytes/token")
        
        return {
            'mean_hidden': mean_hidden,
            'Vt': Vt,
            'S': S,
        }
    
    def test_hybrid_caching(self, n_samples: int = 100):
        """
        Hybrid approach:
        - Use rank-1 for layers 3-26
        - Cache layer 27 output directly (with compression)
        """
        print(f"\n--- Hybrid Caching Test ---")
        print("  (Rank-1 for layers 3-26, cached for layer 27)")
        
        # This would require:
        # 1. Universal directions for layers 3-26 (already have)
        # 2. Scales for layers 3-26 per token
        # 3. Compressed layer 27 output per token
        
        # For now, just estimate storage
        n_rank1_layers = 24  # Layers 3-26
        scales_per_token = n_rank1_layers * 4  # 96 bytes
        
        # Layer 27 output: use k=50 basis
        layer27_coeffs = 50 * 4  # 200 bytes
        
        total_per_token = scales_per_token + layer27_coeffs
        
        print(f"\n  Storage estimate:")
        print(f"    Rank-1 scales (layers 3-26): {scales_per_token} bytes/token")
        print(f"    Layer 27 coefficients (k=50): {layer27_coeffs} bytes/token")
        print(f"    Total: {total_per_token} bytes/token")
        print(f"    Full vocab: {152000 * total_per_token / 1e6:.1f} MB")


def main():
    print("=" * 70)
    print("LAYER 27 ANALYSIS")
    print("=" * 70)
    print("""
Layer 27 has 90% error with rank-1 approximation.
Let's understand why and find an alternative.
""")
    
    analyzer = Layer27Analyzer()
    
    # 1. Analyze true rank
    rank_results = analyzer.analyze_layer27_rank(n_samples=200)
    
    # 2. Analyze structure
    structure_results = analyzer.analyze_layer27_structure(n_samples=100)
    
    # 3. Test direct caching
    cache_results = analyzer.test_direct_caching(n_samples=100)
    
    # 4. Estimate hybrid approach
    analyzer.test_hybrid_caching(n_samples=100)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"""
Layer 27 is NOT rank-1. It requires {rank_results['dims_90']} dimensions for 90% variance.

Options:
1. Cache layer 27 output directly (with basis compression)
2. Use higher-rank approximation for layer 27
3. Hybrid: rank-1 for layers 3-26, cached for layer 27
""")


if __name__ == "__main__":
    main()
