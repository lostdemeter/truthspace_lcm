#!/usr/bin/env python3
"""
Hidden State Basis - Large Scale Test
======================================

Previous finding: 15 basis + 4-bit coeffs = 100% accuracy at 7 bytes/entity
But generalization was only 67% with 12 training samples.

Question: With MORE training data, can we learn a universal basis
that generalizes to new entities?

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class LargeScaleBasisAnalyzer:
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
    
    def test_generalization(self, train_pairs: List[Tuple[str, str]], 
                           test_pairs: List[Tuple[str, str]], 
                           template: str):
        """Test if basis learned from training generalizes to test."""
        
        print(f"\nCollecting {len(train_pairs)} training hidden states...")
        train_hiddens = []
        for entity, _ in train_pairs:
            hidden = self._get_hidden(template.format(entity=entity))
            train_hiddens.append(hidden)
        train_hiddens = np.array(train_hiddens)
        
        print(f"Collecting {len(test_pairs)} test hidden states...")
        test_hiddens = []
        for entity, _ in test_pairs:
            hidden = self._get_hidden(template.format(entity=entity))
            test_hiddens.append(hidden)
        test_hiddens = np.array(test_hiddens)
        
        # Learn basis from training data
        train_mean = train_hiddens.mean(axis=0)
        train_centered = train_hiddens - train_mean
        U_train, S_train, Vt_train = np.linalg.svd(train_centered, full_matrices=False)
        
        print(f"\n--- Training Data Results ---")
        print(f"  Singular values (top 10): {S_train[:10].round(2)}")
        
        # Variance explained
        var_explained = S_train**2 / (S_train**2).sum()
        cumvar = np.cumsum(var_explained)
        print(f"  Variance explained by k=10: {cumvar[min(9, len(cumvar)-1)]*100:.1f}%")
        print(f"  Variance explained by k=20: {cumvar[min(19, len(cumvar)-1)]*100:.1f}%")
        
        # Test on training data
        print(f"\n--- Training Accuracy ---")
        for k in [10, 20, 30, len(S_train)]:
            if k > len(S_train):
                continue
            correct = 0
            for i, hidden in enumerate(train_hiddens):
                centered = hidden - train_mean
                coeffs = centered @ Vt_train[:k, :].T
                reconstructed = train_mean + coeffs @ Vt_train[:k, :]
                
                _, orig_idx = self._decode(hidden)
                _, recon_idx = self._decode(reconstructed)
                if orig_idx == recon_idx:
                    correct += 1
            
            print(f"  k={k}: {correct}/{len(train_hiddens)} = {correct/len(train_hiddens)*100:.0f}%")
        
        # Test on test data (generalization)
        print(f"\n--- Generalization Accuracy ---")
        for k in [10, 20, 30, min(50, len(S_train))]:
            if k > len(S_train):
                continue
            correct = 0
            for i, (entity, _) in enumerate(test_pairs):
                hidden = test_hiddens[i]
                centered = hidden - train_mean
                coeffs = centered @ Vt_train[:k, :].T
                reconstructed = train_mean + coeffs @ Vt_train[:k, :]
                
                orig_token, orig_idx = self._decode(hidden)
                recon_token, recon_idx = self._decode(reconstructed)
                match = orig_idx == recon_idx
                if match:
                    correct += 1
                
                if k == min(50, len(S_train)):
                    status = "✓" if match else "✗"
                    print(f"    {entity}: '{orig_token}' → '{recon_token}' {status}")
            
            accuracy = correct / len(test_pairs)
            storage = k * 1  # 8-bit coefficients
            print(f"  k={k}: {correct}/{len(test_pairs)} = {accuracy*100:.0f}% ({storage} bytes/entity)")
        
        return {
            'train_mean': train_mean,
            'Vt_train': Vt_train,
            'S_train': S_train,
        }


def main():
    print("=" * 70)
    print("LARGE SCALE BASIS TEST")
    print("=" * 70)
    
    analyzer = LargeScaleBasisAnalyzer()
    
    # Larger training set
    train_pairs = [
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
        ("Norway", "Oslo"),
        ("Denmark", "Copenhagen"),
        ("Finland", "Helsinki"),
        ("Poland", "Warsaw"),
        ("Austria", "Vienna"),
        ("Switzerland", "Bern"),
        ("Belgium", "Brussels"),
        ("Netherlands", "Amsterdam"),
        ("Portugal", "Lisbon"),
        ("Ireland", "Dublin"),
        ("Argentina", "Buenos"),
        ("Chile", "Santiago"),
        ("Colombia", "Bogota"),
        ("Peru", "Lima"),
        ("Venezuela", "Caracas"),
    ]
    
    # Test set - entities NOT in training
    test_pairs = [
        ("Thailand", "Bangkok"),
        ("Vietnam", "Hanoi"),
        ("Indonesia", "Jakarta"),
        ("Malaysia", "Kuala"),
        ("Philippines", "Manila"),
        ("Turkey", "Ankara"),
        ("Iran", "Tehran"),
        ("Iraq", "Baghdad"),
        ("Israel", "Jerusalem"),
        ("Kenya", "Nairobi"),
    ]
    
    results = analyzer.test_generalization(
        train_pairs, test_pairs, 
        "The capital of {entity} is"
    )
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key findings:
1. With more training data, we get more basis functions
2. Generalization depends on how well the basis spans the space
3. The question: Is there a UNIVERSAL basis for all entities?
""")


if __name__ == "__main__":
    main()
