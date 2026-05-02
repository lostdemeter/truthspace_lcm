#!/usr/bin/env python3
"""
Hadamard Unwind: The Key to Context?
=====================================

Key finding: h3 = f(emb_A * emb_B) gives 0.963 cosine!

The Hadamard product (element-wise multiplication) captures the relationship.

This is like the MESH in Doc 129:
- MESH = W_q.T @ W_k captures the Q-K relationship
- emb_A * emb_B captures the A-B relationship

Let's explore this further.

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from sklearn.linear_model import Ridge
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2


class HadamardUnwindAnalyzer:
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
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        self.embeddings = self.model.model.embed_tokens.weight.data.float().cpu().numpy()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Vocab size: {self.embeddings.shape[0]}")
    
    def get_layer3_output(self, A: int, B: int) -> np.ndarray:
        """Get layer 3 output for token pair (A, B)."""
        input_ids = torch.tensor([[A, B]]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[4][0, 1].float().cpu().numpy()
    
    def get_final_hidden(self, A: int, B: int) -> np.ndarray:
        """Get final hidden state for token pair (A, B)."""
        input_ids = torch.tensor([[A, B]]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[-1][0, 1].float().cpu().numpy()
    
    def learn_hadamard_transform(self, n_train: int = 500, n_test: int = 100):
        """
        Learn: h3 = W @ (emb_A * emb_B) + b
        
        If this works, we can precompute the transformation and apply it
        to any (A, B) pair using just embeddings!
        """
        print(f"\n--- Learning Hadamard Transform ---")
        print(f"  Training on {n_train} pairs, testing on {n_test} pairs")
        
        # Collect training data
        train_X = []
        train_y = []
        
        print(f"  Collecting training data...")
        for i in range(n_train):
            if i % 100 == 0:
                print(f"    {i}/{n_train}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                hadamard = self.embeddings[A] * self.embeddings[B]
                h3 = self.get_layer3_output(A, B)
                
                train_X.append(hadamard)
                train_y.append(h3)
            except:
                continue
        
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        
        # Learn transformation
        print(f"  Learning transformation W...")
        model = Ridge(alpha=1.0)
        model.fit(train_X, train_y)
        
        # Test on training data
        train_pred = model.predict(train_X)
        train_cos = [np.dot(train_y[i], train_pred[i]) / 
                    (np.linalg.norm(train_y[i]) * np.linalg.norm(train_pred[i]) + 1e-10)
                    for i in range(len(train_y))]
        print(f"  Training cosine: {np.mean(train_cos):.4f}")
        
        # Collect test data
        print(f"  Collecting test data...")
        test_X = []
        test_y = []
        test_pairs = []
        
        for i in range(n_test):
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                hadamard = self.embeddings[A] * self.embeddings[B]
                h3 = self.get_layer3_output(A, B)
                
                test_X.append(hadamard)
                test_y.append(h3)
                test_pairs.append((A, B))
            except:
                continue
        
        test_X = np.array(test_X)
        test_y = np.array(test_y)
        
        # Test
        test_pred = model.predict(test_X)
        test_cos = [np.dot(test_y[i], test_pred[i]) / 
                   (np.linalg.norm(test_y[i]) * np.linalg.norm(test_pred[i]) + 1e-10)
                   for i in range(len(test_y))]
        print(f"  Test cosine: {np.mean(test_cos):.4f}")
        
        return model, test_X, test_y, test_pairs
    
    def test_token_prediction(self, model, n_test: int = 100):
        """
        Test: Can we predict the final token using the Hadamard transform?
        
        Strategy:
        1. Compute h3_pred = W @ (emb_A * emb_B)
        2. Run layers 4-27 from h3_pred
        3. Compare predicted token to actual
        
        For now, just test if h3_pred gives correct token directly via lm_head.
        """
        print(f"\n--- Token Prediction Test ({n_test} pairs) ---")
        
        correct_h3 = 0
        correct_final = 0
        
        for i in range(n_test):
            if i % 20 == 0:
                print(f"  {i}/{n_test}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                # Actual
                h3_actual = self.get_layer3_output(A, B)
                h_final_actual = self.get_final_hidden(A, B)
                true_token = np.argmax(self.lm_head @ h_final_actual)
                
                # Predicted h3
                hadamard = self.embeddings[A] * self.embeddings[B]
                h3_pred = model.predict(hadamard.reshape(1, -1))[0]
                
                # Token from h3 directly (not running layers 4-27)
                token_from_h3_actual = np.argmax(self.lm_head @ h3_actual)
                token_from_h3_pred = np.argmax(self.lm_head @ h3_pred)
                
                if token_from_h3_actual == token_from_h3_pred:
                    correct_h3 += 1
                
                if true_token == token_from_h3_pred:
                    correct_final += 1
                    
            except:
                continue
        
        print(f"\n  Results:")
        print(f"    h3 token match: {correct_h3}/{n_test} = {correct_h3/n_test*100:.1f}%")
        print(f"    Final token match: {correct_final}/{n_test} = {correct_final/n_test*100:.1f}%")
        
        return correct_h3 / n_test, correct_final / n_test
    
    def analyze_transform_structure(self, model):
        """
        Analyze the learned transformation W.
        
        Does it have φ-structure? Is it low-rank?
        """
        print(f"\n--- Analyzing Transform Structure ---")
        
        W = model.coef_  # (hidden_dim, hidden_dim)
        
        print(f"  W shape: {W.shape}")
        print(f"  W norm: {np.linalg.norm(W):.2f}")
        
        # SVD
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        
        # Rank analysis
        total_var = (S**2).sum()
        cumvar = np.cumsum(S**2) / total_var
        
        for threshold in [0.5, 0.9, 0.99]:
            k = np.searchsorted(cumvar, threshold) + 1
            print(f"    {threshold*100:.0f}% variance: k={k}")
        
        # φ-structure in singular values
        print(f"\n  Singular value φ-structure:")
        for i in range(min(10, len(S))):
            level = np.log(S[i]) / np.log(PHI)
            print(f"    S[{i}] = {S[i]:.4f} (φ^{level:.1f})")
        
        return W, S


def main():
    print("=" * 70)
    print("HADAMARD UNWIND: THE KEY TO CONTEXT?")
    print("=" * 70)
    print("""
Key finding: h3 = f(emb_A * emb_B) gives 0.963 cosine!

The Hadamard product captures the A-B relationship.
This is like MESH = W_q.T @ W_k in Doc 129.

Can we learn W such that h3 ≈ W @ (emb_A * emb_B)?
""")
    
    analyzer = HadamardUnwindAnalyzer()
    
    # 1. Learn Hadamard transform
    model, test_X, test_y, test_pairs = analyzer.learn_hadamard_transform(n_train=500, n_test=100)
    
    # 2. Test token prediction
    analyzer.test_token_prediction(model, n_test=100)
    
    # 3. Analyze transform structure
    analyzer.analyze_transform_structure(model)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
