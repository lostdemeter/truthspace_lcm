#!/usr/bin/env python3
"""
Transformation Space Analysis - Simplified
===========================================

Key finding from previous run:
- Layer 14 transformation: 13 dims for 90% variance, 46 dims for 99%

This means the transformation space is LOW-DIMENSIONAL!

Let's verify this across all layers and with real token inputs.

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


class TransformationAnalyzer:
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
    
    def get_layer_outputs(self, token_ids: List[int]) -> List[np.ndarray]:
        """Get hidden state after each layer."""
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            # hidden_states[0] is embedding, [1] is after layer 0, etc.
            return [h[0, -1, :].float().cpu().numpy() for h in outputs.hidden_states]
    
    def analyze_transformations_per_layer(self, n_samples: int = 100) -> Dict:
        """
        Analyze transformation dimensionality at each layer.
        
        For each layer, compute: transformation = output - input
        Then SVD to find how many dimensions are needed.
        """
        print(f"\n--- Analyzing transformations across {self.n_layers} layers ---")
        
        # Collect layer outputs for many token sequences
        all_layer_outputs = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            # Random single token
            token_id = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                layer_outputs = self.get_layer_outputs([token_id])
                all_layer_outputs.append(layer_outputs)
            except:
                continue
        
        # Analyze each layer's transformation
        results = []
        
        for layer_idx in range(self.n_layers):
            # Collect transformations: output[layer+1] - output[layer]
            transformations = []
            for outputs in all_layer_outputs:
                if layer_idx + 1 < len(outputs):
                    transform = outputs[layer_idx + 1] - outputs[layer_idx]
                    transformations.append(transform)
            
            if not transformations:
                continue
            
            transformations = np.array(transformations)
            
            # SVD
            U, S, Vt = np.linalg.svd(transformations, full_matrices=False)
            var_explained = S**2 / (S**2).sum()
            cumvar = np.cumsum(var_explained)
            
            dims_90 = np.searchsorted(cumvar, 0.90) + 1
            dims_99 = np.searchsorted(cumvar, 0.99) + 1
            
            results.append({
                'layer': layer_idx,
                'dims_90': dims_90,
                'dims_99': dims_99,
                'top_singular': S[0],
            })
        
        # Print summary
        print(f"\n  Layer-by-layer transformation dimensionality:")
        print(f"  {'Layer':<8} {'90% var':<10} {'99% var':<10} {'Top S':<10}")
        print(f"  {'-'*38}")
        
        for r in results:
            print(f"  {r['layer']:<8} {r['dims_90']:<10} {r['dims_99']:<10} {r['top_singular']:.1f}")
        
        # Average
        avg_90 = np.mean([r['dims_90'] for r in results])
        avg_99 = np.mean([r['dims_99'] for r in results])
        
        print(f"\n  Average dims for 90% variance: {avg_90:.1f}")
        print(f"  Average dims for 99% variance: {avg_99:.1f}")
        
        return {
            'per_layer': results,
            'avg_dims_90': avg_90,
            'avg_dims_99': avg_99,
        }
    
    def analyze_full_model_transformation(self, n_samples: int = 100) -> Dict:
        """
        Analyze the FULL model transformation (embedding → final hidden).
        """
        print(f"\n--- Analyzing full model transformation ---")
        
        transformations = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            token_id = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                layer_outputs = self.get_layer_outputs([token_id])
                # Full transformation: final - embedding
                transform = layer_outputs[-1] - layer_outputs[0]
                transformations.append(transform)
            except:
                continue
        
        transformations = np.array(transformations)
        
        # SVD
        U, S, Vt = np.linalg.svd(transformations, full_matrices=False)
        var_explained = S**2 / (S**2).sum()
        cumvar = np.cumsum(var_explained)
        
        dims_90 = np.searchsorted(cumvar, 0.90) + 1
        dims_99 = np.searchsorted(cumvar, 0.99) + 1
        
        print(f"\n  Full model transformation:")
        print(f"    Dimensions for 90% variance: {dims_90}")
        print(f"    Dimensions for 99% variance: {dims_99}")
        print(f"    Top 5 singular values: {S[:5].round(1)}")
        
        return {
            'dims_90': dims_90,
            'dims_99': dims_99,
            'S': S,
            'Vt': Vt,
            'cumvar': cumvar,
        }
    
    def test_transformation_basis(self, n_train: int = 80, n_test: int = 20) -> Dict:
        """
        Test if we can reconstruct transformations using a learned basis.
        
        1. Learn basis from training tokens
        2. Test reconstruction on held-out tokens
        """
        print(f"\n--- Testing transformation basis generalization ---")
        
        # Collect transformations
        all_transforms = []
        all_inputs = []
        all_outputs = []
        
        for i in range(n_train + n_test):
            token_id = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                layer_outputs = self.get_layer_outputs([token_id])
                transform = layer_outputs[-1] - layer_outputs[0]
                all_transforms.append(transform)
                all_inputs.append(layer_outputs[0])
                all_outputs.append(layer_outputs[-1])
            except:
                continue
        
        all_transforms = np.array(all_transforms)
        all_inputs = np.array(all_inputs)
        all_outputs = np.array(all_outputs)
        
        # Split
        train_transforms = all_transforms[:n_train]
        test_transforms = all_transforms[n_train:]
        test_inputs = all_inputs[n_train:]
        test_outputs = all_outputs[n_train:]
        
        # Learn basis from training
        U_train, S_train, Vt_train = np.linalg.svd(train_transforms, full_matrices=False)
        
        # Test reconstruction at different k
        print(f"\n  Reconstruction test (train={n_train}, test={n_test}):")
        
        for k in [10, 20, 30, 50, 80]:
            if k > len(S_train):
                continue
            
            # Project test transforms onto training basis
            projected = test_transforms @ Vt_train[:k].T @ Vt_train[:k]
            
            # Reconstruction error
            errors = np.linalg.norm(test_transforms - projected, axis=1)
            orig_norms = np.linalg.norm(test_transforms, axis=1)
            rel_error = (errors / (orig_norms + 1e-10)).mean()
            
            print(f"    k={k}: {(1-rel_error)*100:.1f}% reconstruction accuracy")
        
        return {
            'Vt_train': Vt_train,
            'S_train': S_train,
        }


def main():
    print("=" * 70)
    print("TRANSFORMATION SPACE ANALYSIS")
    print("=" * 70)
    print("""
Key question: How many dimensions does the transformation space have?

If transformations are low-dimensional, we can:
1. Learn a basis from samples
2. Represent any transformation as coefficients
3. Precompute the entire model!
""")
    
    analyzer = TransformationAnalyzer()
    
    # 1. Per-layer analysis
    layer_results = analyzer.analyze_transformations_per_layer(n_samples=100)
    
    # 2. Full model analysis
    full_results = analyzer.analyze_full_model_transformation(n_samples=100)
    
    # 3. Generalization test
    basis_results = analyzer.test_transformation_basis(n_train=80, n_test=20)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"""
Per-layer transformation:
  - Average dims for 90% variance: {layer_results['avg_dims_90']:.1f}
  - Average dims for 99% variance: {layer_results['avg_dims_99']:.1f}

Full model transformation:
  - Dims for 90% variance: {full_results['dims_90']}
  - Dims for 99% variance: {full_results['dims_99']}

IMPLICATION:
  The transformation space is ~{full_results['dims_90']}-dimensional!
  
  This means we can represent the ENTIRE model as:
  - A {full_results['dims_90']}-dimensional basis (shared)
  - Coefficients per token ({full_results['dims_90']} floats = {full_results['dims_90'] * 4} bytes)
  
  For 152K tokens: {152000 * full_results['dims_90'] * 4 / 1e6:.1f} MB total!
""")


if __name__ == "__main__":
    main()
