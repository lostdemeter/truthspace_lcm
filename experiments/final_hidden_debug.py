#!/usr/bin/env python3
"""
Final Hidden Cache Debug
=========================

29% accuracy is too low. Let's understand why.

Hypothesis:
1. Basis learned from 1000 samples doesn't generalize
2. k=100 isn't enough for out-of-sample tokens
3. Need more samples to learn universal basis

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = "/home/thorin/truthspace-lcm/cache/final_hidden"


class FinalHiddenDebugger:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.hidden_dim = self.model.config.hidden_size
        self.vocab_size = self.tokenizer.vocab_size
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        # Load cached data
        self.mean_hidden = np.load(os.path.join(CACHE_DIR, "mean_hidden.npy"))
        self.basis = np.load(os.path.join(CACHE_DIR, "basis.npy"))
        with open(os.path.join(CACHE_DIR, "token_coeffs.pkl"), "rb") as f:
            self.token_coeffs = pickle.load(f)
        
        print(f"  Loaded {len(self.token_coeffs)} cached tokens")
        print(f"  Basis shape: {self.basis.shape}")
    
    def get_final_hidden(self, token_ids):
        input_ids = torch.tensor([token_ids])
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
    
    def test_in_sample_vs_out_of_sample(self, n_tests: int = 50):
        """
        Compare accuracy for tokens used in basis learning vs new tokens.
        """
        print(f"\n--- In-Sample vs Out-of-Sample Test ---")
        
        # In-sample: tokens that were used to learn the basis
        # (We don't have this info, so we'll use cached tokens as proxy)
        
        cached_tokens = list(self.token_coeffs.keys())
        
        # Test on cached tokens (should be high accuracy)
        correct_cached = 0
        for token_id in cached_tokens[:n_tests]:
            true_hidden = self.get_final_hidden([token_id])
            
            coeffs = self.token_coeffs[token_id]
            recon_hidden = self.mean_hidden + coeffs @ self.basis
            
            true_logits = np.dot(self.lm_head, true_hidden)
            recon_logits = np.dot(self.lm_head, recon_hidden)
            
            if np.argmax(true_logits) == np.argmax(recon_logits):
                correct_cached += 1
        
        print(f"  Cached tokens: {correct_cached}/{n_tests} = {correct_cached/n_tests*100:.1f}%")
        
        # Test on NEW tokens (not in cache)
        new_tokens = []
        for i in range(self.vocab_size):
            if i not in self.token_coeffs:
                new_tokens.append(i)
            if len(new_tokens) >= n_tests:
                break
        
        correct_new = 0
        for token_id in new_tokens:
            true_hidden = self.get_final_hidden([token_id])
            
            # Project onto basis
            centered = true_hidden - self.mean_hidden
            coeffs = centered @ self.basis.T
            recon_hidden = self.mean_hidden + coeffs @ self.basis
            
            true_logits = np.dot(self.lm_head, true_hidden)
            recon_logits = np.dot(self.lm_head, recon_hidden)
            
            if np.argmax(true_logits) == np.argmax(recon_logits):
                correct_new += 1
        
        print(f"  New tokens: {correct_new}/{n_tests} = {correct_new/n_tests*100:.1f}%")
    
    def test_different_k(self, n_samples: int = 200):
        """
        Test different k values for basis size.
        """
        print(f"\n--- Testing Different k Values ---")
        
        # Collect fresh samples
        hiddens = []
        sample_tokens = np.random.choice(self.vocab_size, n_samples, replace=False)
        
        for token_id in sample_tokens:
            try:
                hidden = self.get_final_hidden([int(token_id)])
                hiddens.append(hidden)
            except:
                continue
        
        hiddens = np.array(hiddens)
        
        # Split into train/test
        n_train = int(len(hiddens) * 0.8)
        train_hiddens = hiddens[:n_train]
        test_hiddens = hiddens[n_train:]
        
        # Learn basis from training
        train_mean = train_hiddens.mean(axis=0)
        train_centered = train_hiddens - train_mean
        _, S, Vt = np.linalg.svd(train_centered, full_matrices=False)
        
        print(f"\n  Training on {n_train} samples, testing on {len(test_hiddens)}")
        print(f"\n  {'k':<8} {'Train Acc':<12} {'Test Acc':<12} {'Storage':<12}")
        print(f"  {'-'*44}")
        
        for k in [50, 100, 150, 200, 300, 500, n_train-1]:
            if k >= n_train:
                continue
            
            basis = Vt[:k]
            
            # Train accuracy
            train_correct = 0
            for hidden in train_hiddens:
                centered = hidden - train_mean
                coeffs = centered @ basis.T
                recon = train_mean + coeffs @ basis
                
                true_logits = np.dot(self.lm_head, hidden)
                recon_logits = np.dot(self.lm_head, recon)
                
                if np.argmax(true_logits) == np.argmax(recon_logits):
                    train_correct += 1
            
            # Test accuracy
            test_correct = 0
            for hidden in test_hiddens:
                centered = hidden - train_mean
                coeffs = centered @ basis.T
                recon = train_mean + coeffs @ basis
                
                true_logits = np.dot(self.lm_head, hidden)
                recon_logits = np.dot(self.lm_head, recon)
                
                if np.argmax(true_logits) == np.argmax(recon_logits):
                    test_correct += 1
            
            train_acc = train_correct / len(train_hiddens)
            test_acc = test_correct / len(test_hiddens)
            storage = k * 4
            
            print(f"  {k:<8} {train_acc*100:.1f}%{'':<6} {test_acc*100:.1f}%{'':<6} {storage} bytes")
    
    def test_full_hidden_storage(self, n_tests: int = 50):
        """
        What if we just store the FULL hidden state (no compression)?
        This is the upper bound on accuracy.
        """
        print(f"\n--- Full Hidden Storage Test ---")
        
        # For each token, store the full 3584-dim hidden state
        # Storage: 3584 * 4 = 14,336 bytes per token
        
        correct = 0
        test_tokens = np.random.choice(self.vocab_size, n_tests, replace=False)
        
        for token_id in test_tokens:
            hidden = self.get_final_hidden([int(token_id)])
            
            # "Reconstruct" is just the hidden itself
            true_logits = np.dot(self.lm_head, hidden)
            
            # This should always be correct
            correct += 1
        
        print(f"  Full hidden accuracy: {correct}/{n_tests} = 100.0%")
        print(f"  Storage: 14,336 bytes/token = {self.vocab_size * 14336 / 1e9:.2f} GB for full vocab")
    
    def test_quantized_full_hidden(self, n_tests: int = 100):
        """
        What if we quantize the full hidden state?
        """
        print(f"\n--- Quantized Full Hidden Test ---")
        
        test_tokens = np.random.choice(self.vocab_size, n_tests, replace=False)
        
        for bits in [16, 8, 4]:
            correct = 0
            
            for token_id in test_tokens:
                hidden = self.get_final_hidden([int(token_id)])
                
                # Quantize
                max_val = np.abs(hidden).max()
                scale = max_val / (2 ** (bits - 1) - 1)
                quantized = np.round(hidden / scale).astype(int)
                dequantized = quantized * scale
                
                true_logits = np.dot(self.lm_head, hidden)
                quant_logits = np.dot(self.lm_head, dequantized)
                
                if np.argmax(true_logits) == np.argmax(quant_logits):
                    correct += 1
            
            storage = self.hidden_dim * bits // 8
            total_mb = self.vocab_size * storage / 1e6
            
            print(f"  {bits}-bit: {correct}/{n_tests} = {correct/n_tests*100:.1f}% accuracy, {storage} bytes/token, {total_mb:.1f} MB total")


def main():
    print("=" * 70)
    print("FINAL HIDDEN CACHE DEBUG")
    print("=" * 70)
    
    debugger = FinalHiddenDebugger()
    
    # 1. In-sample vs out-of-sample
    debugger.test_in_sample_vs_out_of_sample(n_tests=50)
    
    # 2. Different k values
    debugger.test_different_k(n_samples=300)
    
    # 3. Full hidden storage (upper bound)
    debugger.test_full_hidden_storage(n_tests=50)
    
    # 4. Quantized full hidden
    debugger.test_quantized_full_hidden(n_tests=100)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
