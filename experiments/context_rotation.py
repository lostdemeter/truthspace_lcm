#!/usr/bin/env python3
"""
Context as Rotation in Lattice Space
=====================================

Key insight from Doc 141: The shape is a lattice of 3584 critical lines.
Each hidden state is a point in this space.

Hypothesis: Context ROTATES the point around the lattice.

If h(B) and h(A,B) have cosine similarity 0.65, they're ~49° apart.
This could be a rotation!

The question: Is the rotation CONSISTENT for a given prefix?

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.transform import Rotation
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class ContextRotationAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.hidden_dim = self.model.config.hidden_size
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        print(f"  Hidden dim: {self.hidden_dim}")
    
    def get_final_hidden(self, token_ids: List[int]) -> np.ndarray:
        """Get final hidden state for token sequence."""
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
    
    def compute_rotation_angle(self, h1: np.ndarray, h2: np.ndarray) -> float:
        """Compute angle between two vectors."""
        cos = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-10)
        cos = np.clip(cos, -1, 1)
        return np.arccos(cos) * 180 / np.pi  # degrees
    
    def analyze_rotation_structure(self, n_prefixes: int = 20, n_suffixes: int = 30):
        """
        For each prefix, analyze the rotation it induces.
        
        Key question: Is the rotation angle CONSISTENT across suffixes?
        """
        print(f"\n--- Rotation Structure Analysis ---")
        
        all_angles = []
        prefix_results = []
        
        for p in range(n_prefixes):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            A_text = self.tokenizer.decode([A])
            
            angles = []
            norms_B = []
            norms_AB = []
            
            for s in range(n_suffixes):
                B = np.random.randint(0, self.tokenizer.vocab_size)
                
                try:
                    h_B = self.get_final_hidden([B])
                    h_AB = self.get_final_hidden([A, B])
                    
                    angle = self.compute_rotation_angle(h_B, h_AB)
                    angles.append(angle)
                    norms_B.append(np.linalg.norm(h_B))
                    norms_AB.append(np.linalg.norm(h_AB))
                    all_angles.append(angle)
                except:
                    continue
            
            if len(angles) < 5:
                continue
            
            mean_angle = np.mean(angles)
            std_angle = np.std(angles)
            mean_norm_ratio = np.mean(norms_AB) / np.mean(norms_B)
            
            prefix_results.append({
                'prefix': A_text[:15],
                'mean_angle': mean_angle,
                'std_angle': std_angle,
                'norm_ratio': mean_norm_ratio,
            })
            
            print(f"  Prefix '{A_text[:10]}': angle={mean_angle:.1f}° ± {std_angle:.1f}°, norm_ratio={mean_norm_ratio:.3f}")
        
        # Global statistics
        print(f"\n  Global statistics:")
        print(f"    Mean angle: {np.mean(all_angles):.1f}°")
        print(f"    Std angle: {np.std(all_angles):.1f}°")
        print(f"    Min angle: {np.min(all_angles):.1f}°")
        print(f"    Max angle: {np.max(all_angles):.1f}°")
        
        return prefix_results
    
    def analyze_rotation_plane(self, prefix_token: int, n_suffixes: int = 50):
        """
        For a fixed prefix, find the PLANE of rotation.
        
        If context is a rotation, h(A,B) should lie in a plane defined by:
        - h(B) (the original direction)
        - Some fixed axis (the rotation axis)
        """
        print(f"\n--- Rotation Plane Analysis for prefix {prefix_token} ---")
        
        A = prefix_token
        A_text = self.tokenizer.decode([A])
        print(f"  Prefix: '{A_text}'")
        
        h_B_list = []
        h_AB_list = []
        
        for s in range(n_suffixes):
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_B = self.get_final_hidden([B])
                h_AB = self.get_final_hidden([A, B])
                
                h_B_list.append(h_B)
                h_AB_list.append(h_AB)
            except:
                continue
        
        h_B = np.array(h_B_list)
        h_AB = np.array(h_AB_list)
        
        # Compute the "rotation residual": h(A,B) - projection onto h(B)
        # If rotation, this should lie in a consistent direction
        residuals = []
        for i in range(len(h_B)):
            # Project h_AB onto h_B
            proj = np.dot(h_AB[i], h_B[i]) / (np.dot(h_B[i], h_B[i]) + 1e-10) * h_B[i]
            residual = h_AB[i] - proj
            residuals.append(residual)
        
        residuals = np.array(residuals)
        
        # SVD of residuals - if rotation, should be rank-1 (single axis)
        _, S, Vt = np.linalg.svd(residuals, full_matrices=False)
        
        var_explained = S**2 / (S**2).sum()
        cumvar = np.cumsum(var_explained)
        
        print(f"\n  Residual (h_AB - proj_onto_h_B) analysis:")
        print(f"    Top 5 singular values: {S[:5].round(1)}")
        print(f"    Variance explained by rank-1: {var_explained[0]*100:.1f}%")
        print(f"    Variance explained by rank-5: {cumvar[4]*100:.1f}%")
        
        # The rotation axis is Vt[0]
        rotation_axis = Vt[0]
        
        # Test: Can we reconstruct h_AB using h_B and the rotation axis?
        correct = 0
        for i in range(len(h_B)):
            # Reconstruct: h_AB ≈ a * h_B + b * rotation_axis
            # Solve for a, b
            A_mat = np.column_stack([h_B[i], rotation_axis])
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, h_AB[i], rcond=None)
            
            reconstructed = coeffs[0] * h_B[i] + coeffs[1] * rotation_axis
            
            # Decode
            true_logits = np.dot(self.lm_head, h_AB[i])
            recon_logits = np.dot(self.lm_head, reconstructed)
            
            if np.argmax(true_logits) == np.argmax(recon_logits):
                correct += 1
        
        accuracy = correct / len(h_B)
        print(f"\n  Reconstruction accuracy (h_B + rotation_axis): {accuracy*100:.1f}%")
        
        return {
            'rotation_axis': rotation_axis,
            'var_explained_rank1': var_explained[0],
            'accuracy': accuracy,
        }
    
    def test_universal_rotation_axis(self, n_samples: int = 100):
        """
        Is there a UNIVERSAL rotation axis that works for all prefixes?
        """
        print(f"\n--- Universal Rotation Axis Test ---")
        
        # Collect many (h_B, h_AB) pairs
        h_B_list = []
        h_AB_list = []
        
        for i in range(n_samples):
            if i % 20 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_B = self.get_final_hidden([B])
                h_AB = self.get_final_hidden([A, B])
                
                h_B_list.append(h_B)
                h_AB_list.append(h_AB)
            except:
                continue
        
        h_B = np.array(h_B_list)
        h_AB = np.array(h_AB_list)
        
        # Compute residuals
        residuals = []
        for i in range(len(h_B)):
            proj = np.dot(h_AB[i], h_B[i]) / (np.dot(h_B[i], h_B[i]) + 1e-10) * h_B[i]
            residual = h_AB[i] - proj
            residuals.append(residual)
        
        residuals = np.array(residuals)
        
        # SVD
        _, S, Vt = np.linalg.svd(residuals, full_matrices=False)
        
        var_explained = S**2 / (S**2).sum()
        cumvar = np.cumsum(var_explained)
        
        print(f"\n  Universal residual analysis:")
        print(f"    Top 10 singular values: {S[:10].round(1)}")
        print(f"    Dims for 50% variance: {np.searchsorted(cumvar, 0.50) + 1}")
        print(f"    Dims for 90% variance: {np.searchsorted(cumvar, 0.90) + 1}")
        print(f"    Dims for 99% variance: {np.searchsorted(cumvar, 0.99) + 1}")
        
        # Test reconstruction with different ranks
        print(f"\n  Reconstruction accuracy with universal basis:")
        
        for k in [1, 5, 10, 20, 50]:
            basis = Vt[:k]
            correct = 0
            
            for i in range(len(h_B)):
                # Reconstruct: h_AB ≈ proj_onto_h_B + proj_onto_basis
                proj_B = np.dot(h_AB[i], h_B[i]) / (np.dot(h_B[i], h_B[i]) + 1e-10) * h_B[i]
                residual = h_AB[i] - proj_B
                
                # Project residual onto basis
                coeffs = residual @ basis.T
                recon_residual = coeffs @ basis
                
                reconstructed = proj_B + recon_residual
                
                # Decode
                true_logits = np.dot(self.lm_head, h_AB[i])
                recon_logits = np.dot(self.lm_head, reconstructed)
                
                if np.argmax(true_logits) == np.argmax(recon_logits):
                    correct += 1
            
            accuracy = correct / len(h_B)
            print(f"    Rank-{k}: {accuracy*100:.1f}%")


def main():
    print("=" * 70)
    print("CONTEXT AS ROTATION IN LATTICE SPACE")
    print("=" * 70)
    print("""
Hypothesis: Context ROTATES the hidden state in the lattice.

If true:
- h(A,B) = R(A) @ h(B) where R(A) is a rotation matrix
- The rotation depends on prefix A but is consistent across suffixes B
- We can precompute R(A) for each prefix token
""")
    
    analyzer = ContextRotationAnalyzer()
    
    # 1. Analyze rotation angles
    rotation_results = analyzer.analyze_rotation_structure(n_prefixes=20, n_suffixes=30)
    
    # 2. Analyze rotation plane for a specific prefix
    sample_prefix = np.random.randint(0, analyzer.tokenizer.vocab_size)
    plane_results = analyzer.analyze_rotation_plane(sample_prefix, n_suffixes=50)
    
    # 3. Test universal rotation axis
    analyzer.test_universal_rotation_axis(n_samples=100)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
