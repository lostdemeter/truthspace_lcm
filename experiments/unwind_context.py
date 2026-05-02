#!/usr/bin/env python3
"""
Unwind Context: Rearrange Data Geometrically
==============================================

Key insight from user:
> "It's possible that we need to unwind the transformer, and rearrange 
>  the data so that it makes geometric sense for our own purposes"

The transformer arranged data for ITS purposes (gradient descent optimization).
We need to rearrange it for OURS (geometric computation).

From Doc 129: Unraveling MESH = W_q.T @ W_k eliminates error compounding.
From Doc 151: Model is just indices into 92-entry LUT.

Question: Can we unwind the (A,B) → h3 mapping into a geometric format?

Hypothesis: The "hash table" is actually a geometric structure in disguise.
If we can find the right basis, the mapping might become simple.

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2


class UnwindContextAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        config = AutoConfig.from_pretrained(model_name)
        config._attn_implementation = "eager"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = next(self.model.parameters()).device
        
        self.hidden_dim = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        # Get embeddings
        self.embeddings = self.model.model.embed_tokens.weight.data.float().cpu().numpy()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Vocab size: {self.embeddings.shape[0]}")
    
    def get_layer3_output(self, A: int, B: int) -> np.ndarray:
        """Get layer 3 output for token pair (A, B)."""
        input_ids = torch.tensor([[A, B]]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[4][0, 1].float().cpu().numpy()
    
    def analyze_embedding_relationship(self, n_samples: int = 500):
        """
        Analyze: Is h3(A,B) related to emb(A) and emb(B) in a geometric way?
        
        Maybe h3(A,B) = emb(A) ⊗ emb(B) in some transformed space?
        """
        print(f"\n--- Analyzing Embedding Relationship ({n_samples} pairs) ---")
        
        # Collect data
        emb_A_list = []
        emb_B_list = []
        h3_list = []
        
        for i in range(n_samples):
            if i % 100 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                emb_A = self.embeddings[A]
                emb_B = self.embeddings[B]
                h3 = self.get_layer3_output(A, B)
                
                emb_A_list.append(emb_A)
                emb_B_list.append(emb_B)
                h3_list.append(h3)
            except:
                continue
        
        emb_A = np.array(emb_A_list)
        emb_B = np.array(emb_B_list)
        h3 = np.array(h3_list)
        
        print(f"\n  Collected {len(h3)} samples")
        
        # Test 1: Is h3 in the span of emb_A and emb_B?
        print(f"\n  Test 1: h3 = w_A * emb_A + w_B * emb_B?")
        
        residuals = []
        for i in range(len(h3)):
            X = np.column_stack([emb_A[i], emb_B[i]])
            coeffs, _, _, _ = np.linalg.lstsq(X, h3[i], rcond=None)
            pred = coeffs[0] * emb_A[i] + coeffs[1] * emb_B[i]
            residual = np.linalg.norm(h3[i] - pred) / np.linalg.norm(h3[i])
            residuals.append(residual)
        
        print(f"    Mean residual: {np.mean(residuals):.4f}")
        print(f"    (0 = perfect, 1 = no relationship)")
        
        # Test 2: Is h3 related to emb_A * emb_B (Hadamard)?
        print(f"\n  Test 2: h3 = f(emb_A * emb_B)?")
        
        hadamard = emb_A * emb_B
        
        # Linear regression
        coeffs, _, _, _ = np.linalg.lstsq(hadamard, h3, rcond=None)
        pred = hadamard @ coeffs
        
        cos_sims = [np.dot(h3[i], pred[i]) / (np.linalg.norm(h3[i]) * np.linalg.norm(pred[i]) + 1e-10)
                   for i in range(len(h3))]
        print(f"    Mean cosine: {np.mean(cos_sims):.4f}")
        
        # Test 3: Is h3 related to emb_A ⊕ emb_B (concatenation)?
        print(f"\n  Test 3: h3 = f(concat(emb_A, emb_B))?")
        
        concat = np.concatenate([emb_A, emb_B], axis=1)  # (n, 2*hidden)
        
        # PCA to reduce dimension
        pca = PCA(n_components=min(100, len(h3)-1))
        concat_reduced = pca.fit_transform(concat)
        
        # Linear regression
        coeffs, _, _, _ = np.linalg.lstsq(concat_reduced, h3, rcond=None)
        pred = concat_reduced @ coeffs
        
        cos_sims = [np.dot(h3[i], pred[i]) / (np.linalg.norm(h3[i]) * np.linalg.norm(pred[i]) + 1e-10)
                   for i in range(len(h3))]
        print(f"    Mean cosine: {np.mean(cos_sims):.4f}")
        
        return {
            'emb_A': emb_A,
            'emb_B': emb_B,
            'h3': h3,
        }
    
    def analyze_phi_structure(self, h3: np.ndarray):
        """
        Analyze: Does h3 have φ-structure?
        
        From Doc 151: Model is just indices into 92-entry LUT.
        Maybe h3 values cluster on φ-levels?
        """
        print(f"\n--- Analyzing φ-Structure in h3 ---")
        
        # Flatten all h3 values
        all_values = h3.flatten()
        
        # Compute φ-levels
        signs = np.sign(all_values)
        magnitudes = np.abs(all_values)
        magnitudes = np.clip(magnitudes, 1e-10, None)  # Avoid log(0)
        
        levels = np.round(np.log(magnitudes) / np.log(PHI))
        
        # Distribution of levels
        unique_levels, counts = np.unique(levels, return_counts=True)
        
        print(f"  Level distribution:")
        print(f"    Min level: {levels.min():.0f}")
        print(f"    Max level: {levels.max():.0f}")
        print(f"    Mean level: {levels.mean():.1f}")
        
        # Top 10 levels
        top_indices = np.argsort(counts)[-10:][::-1]
        print(f"\n  Top 10 levels:")
        for idx in top_indices:
            level = unique_levels[idx]
            count = counts[idx]
            pct = count / len(levels) * 100
            print(f"    Level {level:4.0f}: {count:8d} ({pct:.1f}%)")
        
        # Reconstruction error using φ-quantization
        reconstructed = signs * (PHI ** levels)
        error = np.abs(all_values - reconstructed).mean()
        baseline = np.abs(all_values).mean()
        
        print(f"\n  φ-quantization error: {error:.6f} (baseline: {baseline:.4f})")
        print(f"  Relative error: {error/baseline*100:.2f}%")
        
        return levels
    
    def test_geometric_unwind(self, n_samples: int = 200):
        """
        Test: Can we "unwind" the (A,B) → h3 mapping?
        
        Hypothesis: There exists a transformation T such that:
        T(h3(A,B)) = T(emb(A)) ⊗ T(emb(B))
        
        Where ⊗ is some simple operation (addition, Hadamard, etc.)
        """
        print(f"\n--- Testing Geometric Unwind ({n_samples} pairs) ---")
        
        # Collect data
        emb_A_list = []
        emb_B_list = []
        h3_list = []
        
        for i in range(n_samples):
            if i % 50 == 0:
                print(f"  Collecting {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                emb_A_list.append(self.embeddings[A])
                emb_B_list.append(self.embeddings[B])
                h3_list.append(self.get_layer3_output(A, B))
            except:
                continue
        
        emb_A = np.array(emb_A_list)
        emb_B = np.array(emb_B_list)
        h3 = np.array(h3_list)
        
        # Learn transformation T_emb for embeddings
        # Learn transformation T_h3 for h3
        # Such that T_h3(h3) ≈ T_emb(emb_A) + T_emb(emb_B)
        
        print(f"\n  Learning geometric transformation...")
        
        # Use PCA to find principal directions
        pca_emb = PCA(n_components=50)
        pca_h3 = PCA(n_components=50)
        
        emb_A_reduced = pca_emb.fit_transform(emb_A)
        emb_B_reduced = pca_emb.transform(emb_B)
        h3_reduced = pca_h3.fit_transform(h3)
        
        # Test: h3_reduced ≈ W @ (emb_A_reduced + emb_B_reduced)
        emb_sum = emb_A_reduced + emb_B_reduced
        
        W, _, _, _ = np.linalg.lstsq(emb_sum, h3_reduced, rcond=None)
        pred = emb_sum @ W
        
        cos_sims = [np.dot(h3_reduced[i], pred[i]) / 
                   (np.linalg.norm(h3_reduced[i]) * np.linalg.norm(pred[i]) + 1e-10)
                   for i in range(len(h3_reduced))]
        
        print(f"    h3 ≈ W @ (emb_A + emb_B): mean cosine = {np.mean(cos_sims):.4f}")
        
        # Test: h3_reduced ≈ W @ (emb_A_reduced * emb_B_reduced)
        emb_prod = emb_A_reduced * emb_B_reduced
        
        W, _, _, _ = np.linalg.lstsq(emb_prod, h3_reduced, rcond=None)
        pred = emb_prod @ W
        
        cos_sims = [np.dot(h3_reduced[i], pred[i]) / 
                   (np.linalg.norm(h3_reduced[i]) * np.linalg.norm(pred[i]) + 1e-10)
                   for i in range(len(h3_reduced))]
        
        print(f"    h3 ≈ W @ (emb_A * emb_B): mean cosine = {np.mean(cos_sims):.4f}")
        
        # Test: Bilinear form h3 ≈ emb_A @ M @ emb_B.T
        # This is like the MESH in Doc 129!
        print(f"\n  Testing bilinear form (like MESH)...")
        
        # For each dimension of h3, learn a bilinear form
        # h3[d] ≈ emb_A @ M_d @ emb_B.T
        
        # This is expensive, so just test a few dimensions
        bilinear_cos = []
        for d in range(min(10, h3.shape[1])):
            # Solve for M_d such that h3[:, d] ≈ sum_i sum_j emb_A[:, i] * M_d[i,j] * emb_B[:, j]
            # This is a rank-1 approximation problem
            
            # Simplified: h3[:, d] ≈ (emb_A @ v) * (emb_B @ w) for some v, w
            # Use SVD on the outer product structure
            
            # Actually, let's just test if h3[d] correlates with emb_A @ emb_B.T diagonal
            diag = np.sum(emb_A * emb_B, axis=1)  # (n,)
            corr = np.corrcoef(h3[:, d], diag)[0, 1]
            bilinear_cos.append(abs(corr))
        
        print(f"    Correlation with emb_A · emb_B: {np.mean(bilinear_cos):.4f}")
        
        return {
            'emb_A': emb_A,
            'emb_B': emb_B,
            'h3': h3,
        }


def main():
    print("=" * 70)
    print("UNWIND CONTEXT: REARRANGE DATA GEOMETRICALLY")
    print("=" * 70)
    print("""
The transformer arranged data for ITS purposes.
We need to rearrange it for OURS.

From Doc 129: Unraveling MESH = W_q.T @ W_k eliminates error compounding.
From Doc 151: Model is just indices into 92-entry LUT.

Question: Can we unwind (A,B) → h3 into a geometric format?
""")
    
    analyzer = UnwindContextAnalyzer()
    
    # 1. Analyze embedding relationship
    data = analyzer.analyze_embedding_relationship(n_samples=300)
    
    # 2. Analyze φ-structure
    analyzer.analyze_phi_structure(data['h3'])
    
    # 3. Test geometric unwind
    analyzer.test_geometric_unwind(n_samples=200)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
