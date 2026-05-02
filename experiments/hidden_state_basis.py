#!/usr/bin/env python3
"""
Hidden State Basis Decomposition
=================================

From Bulge Discovery (Doc 180):
- Trajectories = Geodesic + Bulge
- Bulge shape is UNIVERSAL
- Only 10 coefficients capture 87.5% variance
- 2,867x compression!

Hypothesis: Hidden states might have similar structure.
- Shared basis functions (like bulge wavelets)
- Entity-specific coefficients (like bulge coefficients)

If H = Σ c_i × ψ_i where ψ_i are shared basis functions,
we only need to store the coefficients c_i per entity.

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class HiddenStateBasisAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        self.hidden_dim = self.model.config.hidden_size
        
        print(f"  Hidden dim: {self.hidden_dim}")
    
    def _get_hidden(self, prompt: str) -> np.ndarray:
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
    
    def _decode(self, hidden: np.ndarray) -> Tuple[str, int]:
        logits = np.dot(self.lm_head, hidden)
        idx = np.argmax(logits)
        return self.tokenizer.decode([idx]).strip(), idx
    
    def analyze_basis(self, pairs: List[Tuple[str, str]], template: str):
        """Analyze basis decomposition of hidden states."""
        print(f"\nCollecting {len(pairs)} hidden states...")
        
        hiddens = []
        entities = []
        
        for entity, answer in pairs:
            hidden = self._get_hidden(template.format(entity=entity))
            hiddens.append(hidden)
            entities.append(entity)
        
        hiddens = np.array(hiddens)
        
        # 1. SVD to find basis functions
        print("\n--- SVD Basis Analysis ---")
        
        # Center the data
        mean_hidden = hiddens.mean(axis=0)
        centered = hiddens - mean_hidden
        
        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # U: (n_samples, n_components) - coefficients
        # S: singular values
        # Vt: (n_components, hidden_dim) - basis functions
        
        print(f"  Singular values (top 10): {S[:10].round(2)}")
        
        # Variance explained
        var_explained = S**2 / (S**2).sum()
        cumvar = np.cumsum(var_explained)
        
        print(f"\n  Variance explained:")
        for k in [1, 2, 3, 5, 10]:
            if k <= len(S):
                print(f"    k={k}: {cumvar[k-1]*100:.1f}%")
        
        # 2. Test reconstruction with k basis functions
        print("\n--- Basis Reconstruction Test ---")
        
        for k in [1, 2, 3, 5, 10, len(S)]:
            if k > len(S):
                continue
            
            correct = 0
            for i in range(len(hiddens)):
                # Reconstruct: mean + Σ c_j × ψ_j for j=1..k
                # c_j = U[i, j] * S[j]
                # ψ_j = Vt[j, :]
                coeffs = U[i, :k] * S[:k]
                basis = Vt[:k, :]
                
                reconstructed = mean_hidden + coeffs @ basis
                
                _, orig_idx = self._decode(hiddens[i])
                _, recon_idx = self._decode(reconstructed)
                if orig_idx == recon_idx:
                    correct += 1
            
            # Storage: k coefficients (float32)
            storage = k * 4
            accuracy = correct / len(hiddens)
            print(f"  k={k}: {accuracy*100:.0f}% accuracy, {storage} bytes/entity")
        
        # 3. Test with quantized coefficients
        print("\n--- Quantized Coefficients ---")
        
        k = len(S)  # Use all components first
        
        for bits in [16, 8, 4]:
            correct = 0
            for i in range(len(hiddens)):
                coeffs = U[i, :k] * S[:k]
                
                # Quantize coefficients
                max_coeff = np.abs(coeffs).max()
                scale = max_coeff / (2 ** (bits - 1) - 1)
                quantized = np.round(coeffs / scale).astype(int)
                dequantized = quantized * scale
                
                reconstructed = mean_hidden + dequantized @ Vt[:k, :]
                
                _, orig_idx = self._decode(hiddens[i])
                _, recon_idx = self._decode(reconstructed)
                if orig_idx == recon_idx:
                    correct += 1
            
            storage = k * bits // 8
            accuracy = correct / len(hiddens)
            print(f"  {bits}-bit coeffs (k={k}): {accuracy*100:.0f}% accuracy, {storage} bytes/entity")
        
        # 4. The key question: Can we LEARN the basis from MORE data?
        print("\n--- Generalization Test ---")
        print("  (Using basis learned from training data on new entities)")
        
        # Split into train/test
        n_train = len(pairs) - 3
        train_hiddens = hiddens[:n_train]
        test_hiddens = hiddens[n_train:]
        test_entities = entities[n_train:]
        
        # Learn basis from training data
        train_mean = train_hiddens.mean(axis=0)
        train_centered = train_hiddens - train_mean
        U_train, S_train, Vt_train = np.linalg.svd(train_centered, full_matrices=False)
        
        # Project test data onto training basis
        for k in [5, 10, min(n_train-1, 14)]:
            if k > len(S_train):
                continue
            
            correct = 0
            for i, hidden in enumerate(test_hiddens):
                # Project onto basis: c = (h - mean) @ V.T
                centered_test = hidden - train_mean
                coeffs = centered_test @ Vt_train[:k, :].T
                
                # Reconstruct
                reconstructed = train_mean + coeffs @ Vt_train[:k, :]
                
                _, orig_idx = self._decode(hidden)
                _, recon_idx = self._decode(reconstructed)
                if orig_idx == recon_idx:
                    correct += 1
                
                print(f"    {test_entities[i]}: orig='{self._decode(hidden)[0]}' recon='{self._decode(reconstructed)[0]}'")
            
            accuracy = correct / len(test_hiddens)
            print(f"  k={k}: {accuracy*100:.0f}% generalization accuracy")
        
        # 5. What if we use the LM head rows as basis?
        print("\n--- LM Head as Basis ---")
        print("  (The answer token's LM head row might be the 'direction' we need)")
        
        # Get answer token IDs
        answer_ids = []
        for _, answer in pairs:
            ids = self.tokenizer.encode(answer, add_special_tokens=False)
            answer_ids.append(ids[0] if ids else -1)
        
        # For each hidden state, what's its projection onto its answer's LM head row?
        projections = []
        for i, hidden in enumerate(hiddens):
            lm_row = self.lm_head[answer_ids[i]]
            proj = np.dot(hidden, lm_row) / (np.linalg.norm(lm_row) ** 2)
            projections.append(proj)
        
        print(f"  Projection onto answer LM head:")
        print(f"    Mean: {np.mean(projections):.3f}")
        print(f"    Std: {np.std(projections):.3f}")
        
        # Test: Can we reconstruct using just the answer direction?
        print("\n  Reconstruction using answer LM head direction:")
        
        for scale in [1.0, 5.0, 10.0, 50.0, 100.0]:
            correct = 0
            for i, hidden in enumerate(hiddens):
                lm_row = self.lm_head[answer_ids[i]]
                lm_row_norm = lm_row / np.linalg.norm(lm_row)
                
                reconstructed = mean_hidden + scale * lm_row_norm
                
                _, orig_idx = self._decode(hidden)
                _, recon_idx = self._decode(reconstructed)
                if orig_idx == recon_idx:
                    correct += 1
            
            accuracy = correct / len(hiddens)
            print(f"    scale={scale}: {accuracy*100:.0f}% accuracy")
        
        return {
            'hiddens': hiddens,
            'mean_hidden': mean_hidden,
            'U': U,
            'S': S,
            'Vt': Vt,
        }


def main():
    print("=" * 70)
    print("HIDDEN STATE BASIS DECOMPOSITION")
    print("=" * 70)
    print("""
From Bulge Discovery:
- Trajectories = Geodesic + Bulge
- Bulge = Σ c_i × ψ_i (10 coefficients, 87.5% variance)
- 2,867x compression!

Question: Can hidden states be decomposed similarly?
H = mean + Σ c_i × ψ_i

If so, we only store coefficients per entity.
""")
    
    analyzer = HiddenStateBasisAnalyzer()
    
    pairs = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
        ("Japan", "Tokyo"),
        ("China", "Beijing"),
        ("India", "Delhi"),
        ("Brazil", "Brasilia"),
        ("Canada", "Ottawa"),
        ("Australia", "Canberra"),
        ("Russia", "Moscow"),
        ("Mexico", "Mexico"),
        ("Egypt", "Cairo"),
        ("Greece", "Athens"),
        ("Sweden", "Stockholm"),
    ]
    
    results = analyzer.analyze_basis(pairs, "The capital of {entity} is")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    main()
