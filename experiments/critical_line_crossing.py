#!/usr/bin/env python3
"""
Critical Line Crossing Analysis
================================

From Doc 141: The shape is 3584 critical lines.
Each hidden state is defined by which side of each line it's on.

When we add context A to suffix B:
- Some critical lines get crossed (sign flips)
- The pattern of crossings depends on BOTH A and B

Key question: Can we predict WHICH lines get crossed?

If we can predict the crossing pattern from h(A) and h(B),
we can compute h(A,B) = h(B) * crossing_pattern

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class CriticalLineCrossingAnalyzer:
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
    
    def compute_crossing_pattern(self, h_B: np.ndarray, h_AB: np.ndarray) -> np.ndarray:
        """
        Compute which critical lines were crossed.
        
        A crossing occurs when sign(h_B[i]) != sign(h_AB[i])
        
        Returns: +1 if same side, -1 if crossed
        """
        signs_B = np.sign(h_B)
        signs_AB = np.sign(h_AB)
        
        # +1 if same sign, -1 if different
        crossing = signs_B * signs_AB
        
        return crossing
    
    def analyze_crossing_predictability(self, n_samples: int = 200):
        """
        Can we predict the crossing pattern from h(A) and h(B)?
        
        If crossing[i] = f(h_A[i], h_B[i]), we can precompute!
        """
        print(f"\n--- Crossing Predictability Analysis ({n_samples} pairs) ---")
        
        h_A_list = []
        h_B_list = []
        crossings_list = []
        
        for i in range(n_samples):
            if i % 50 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_A = self.get_final_hidden([A])
                h_B = self.get_final_hidden([B])
                h_AB = self.get_final_hidden([A, B])
                
                crossing = self.compute_crossing_pattern(h_B, h_AB)
                
                h_A_list.append(h_A)
                h_B_list.append(h_B)
                crossings_list.append(crossing)
            except:
                continue
        
        h_A = np.array(h_A_list)
        h_B = np.array(h_B_list)
        crossings = np.array(crossings_list)
        
        # For each dimension, can we predict whether it crosses?
        print(f"\n  Per-dimension crossing analysis:")
        
        # Simple predictor: crossing[i] = sign(h_A[i] * h_B[i])
        # (If A and B have same sign, no crossing; if opposite, crossing)
        pred_simple = np.sign(h_A * h_B)
        accuracy_simple = (pred_simple == crossings).mean()
        print(f"    Simple predictor (sign(h_A * h_B)): {accuracy_simple*100:.1f}%")
        
        # Predictor based on sign of h_A only
        pred_A = np.sign(h_A)
        accuracy_A = (pred_A == crossings).mean()
        print(f"    Sign of h_A: {accuracy_A*100:.1f}%")
        
        # Predictor based on sign of h_B only
        pred_B = np.sign(h_B)
        accuracy_B = (pred_B == crossings).mean()
        print(f"    Sign of h_B: {accuracy_B*100:.1f}%")
        
        # Random baseline
        print(f"    Random baseline: 50.0%")
        
        # Per-dimension analysis: which dimensions are predictable?
        dim_accuracies = []
        for d in range(self.hidden_dim):
            # For this dimension, what predicts crossing?
            # Try: crossing = sign(h_A[d]) * sign(h_B[d])
            pred = np.sign(h_A[:, d]) * np.sign(h_B[:, d])
            acc = (pred == crossings[:, d]).mean()
            dim_accuracies.append(acc)
        
        dim_accuracies = np.array(dim_accuracies)
        
        print(f"\n  Per-dimension accuracy distribution:")
        print(f"    Mean: {dim_accuracies.mean()*100:.1f}%")
        print(f"    Std: {dim_accuracies.std()*100:.1f}%")
        print(f"    Min: {dim_accuracies.min()*100:.1f}%")
        print(f"    Max: {dim_accuracies.max()*100:.1f}%")
        print(f"    Dims > 70%: {(dim_accuracies > 0.7).sum()}")
        print(f"    Dims > 80%: {(dim_accuracies > 0.8).sum()}")
        print(f"    Dims > 90%: {(dim_accuracies > 0.9).sum()}")
        
        return {
            'h_A': h_A,
            'h_B': h_B,
            'crossings': crossings,
            'dim_accuracies': dim_accuracies,
        }
    
    def test_crossing_based_reconstruction(self, n_train: int = 150, n_test: int = 50):
        """
        Test: Can we reconstruct h(A,B) using predicted crossings?
        
        h(A,B) ≈ h(B) * predicted_crossing
        """
        print(f"\n--- Crossing-Based Reconstruction Test ---")
        
        # Collect data
        h_A_list = []
        h_B_list = []
        h_AB_list = []
        crossings_list = []
        
        for i in range(n_train + n_test):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_A = self.get_final_hidden([A])
                h_B = self.get_final_hidden([B])
                h_AB = self.get_final_hidden([A, B])
                
                crossing = self.compute_crossing_pattern(h_B, h_AB)
                
                h_A_list.append(h_A)
                h_B_list.append(h_B)
                h_AB_list.append(h_AB)
                crossings_list.append(crossing)
            except:
                continue
        
        h_A = np.array(h_A_list)
        h_B = np.array(h_B_list)
        h_AB = np.array(h_AB_list)
        crossings = np.array(crossings_list)
        
        # Split
        h_A_train, h_A_test = h_A[:n_train], h_A[n_train:]
        h_B_train, h_B_test = h_B[:n_train], h_B[n_train:]
        h_AB_train, h_AB_test = h_AB[:n_train], h_AB[n_train:]
        crossings_train, crossings_test = crossings[:n_train], crossings[n_train:]
        
        # Learn per-dimension crossing predictor
        # For each dimension d: crossing[d] = sign(w_A[d] * h_A[d] + w_B[d] * h_B[d])
        
        # Simple approach: crossing[d] = sign(h_A[d] * h_B[d])
        pred_crossings_test = np.sign(h_A_test * h_B_test)
        
        # Reconstruct h_AB using predicted crossings
        # h_AB ≈ h_B * crossing (just flip signs)
        reconstructed = h_B_test * pred_crossings_test
        
        # But we also need to adjust magnitudes...
        # Let's try: h_AB ≈ |h_AB_mean| * sign(h_B * crossing)
        
        # Actually, let's just test sign accuracy
        sign_accuracy = (np.sign(reconstructed) == np.sign(h_AB_test)).mean()
        print(f"\n  Sign accuracy (predicted vs true): {sign_accuracy*100:.1f}%")
        
        # Token prediction accuracy
        correct = 0
        for i in range(len(h_AB_test)):
            true_token = np.argmax(self.lm_head @ h_AB_test[i])
            pred_token = np.argmax(self.lm_head @ reconstructed[i])
            if true_token == pred_token:
                correct += 1
        
        accuracy = correct / len(h_AB_test)
        print(f"  Token prediction accuracy: {correct}/{len(h_AB_test)} = {accuracy*100:.1f}%")
        
        # What if we use TRUE crossings but predicted magnitudes?
        # h_AB ≈ |h_B| * true_crossing
        reconstructed_true_cross = np.abs(h_B_test) * crossings_test
        
        correct_true_cross = 0
        for i in range(len(h_AB_test)):
            true_token = np.argmax(self.lm_head @ h_AB_test[i])
            pred_token = np.argmax(self.lm_head @ reconstructed_true_cross[i])
            if true_token == pred_token:
                correct_true_cross += 1
        
        accuracy_true_cross = correct_true_cross / len(h_AB_test)
        print(f"  Token accuracy with TRUE crossings: {correct_true_cross}/{len(h_AB_test)} = {accuracy_true_cross*100:.1f}%")
        
        return {
            'sign_accuracy': sign_accuracy,
            'token_accuracy': accuracy,
            'token_accuracy_true_cross': accuracy_true_cross,
        }
    
    def analyze_magnitude_structure(self, n_samples: int = 100):
        """
        Separate analysis of magnitudes.
        
        If signs are the critical lines, magnitudes are the DISTANCE from each line.
        How do magnitudes change with context?
        """
        print(f"\n--- Magnitude Structure Analysis ---")
        
        mag_ratios = []
        
        for i in range(n_samples):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                h_B = self.get_final_hidden([B])
                h_AB = self.get_final_hidden([A, B])
                
                # Per-dimension magnitude ratio
                ratio = np.abs(h_AB) / (np.abs(h_B) + 1e-10)
                mag_ratios.append(ratio)
            except:
                continue
        
        mag_ratios = np.array(mag_ratios)
        
        # Statistics
        mean_ratio = mag_ratios.mean(axis=0)
        std_ratio = mag_ratios.std(axis=0)
        
        print(f"\n  Magnitude ratio |h_AB| / |h_B|:")
        print(f"    Global mean: {mean_ratio.mean():.3f}")
        print(f"    Global std: {mean_ratio.std():.3f}")
        print(f"    Per-dim std mean: {std_ratio.mean():.3f}")
        
        # Is the ratio consistent per dimension?
        consistent_dims = (std_ratio < 0.5).sum()
        print(f"    Consistent dims (std < 0.5): {consistent_dims}")
        
        return {
            'mean_ratio': mean_ratio,
            'std_ratio': std_ratio,
        }


def main():
    print("=" * 70)
    print("CRITICAL LINE CROSSING ANALYSIS")
    print("=" * 70)
    print("""
From Doc 141: The shape is 3584 critical lines.
Adding context crosses some lines (flips signs).

Key question: Can we predict WHICH lines get crossed?
If yes, we can compute h(A,B) = h(B) * crossing_pattern
""")
    
    analyzer = CriticalLineCrossingAnalyzer()
    
    # 1. Crossing predictability
    crossing_results = analyzer.analyze_crossing_predictability(n_samples=200)
    
    # 2. Crossing-based reconstruction
    recon_results = analyzer.test_crossing_based_reconstruction(n_train=150, n_test=50)
    
    # 3. Magnitude structure
    mag_results = analyzer.analyze_magnitude_structure(n_samples=100)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
