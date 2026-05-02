#!/usr/bin/env python3
"""
Hidden State Structure: Simplified Analysis
=============================================

Key question: Can we decompose hidden_state = f(entity_embedding) + residual?

If the residual is SHARED across entities (relationship-specific),
we can compute hidden states without the transformer.

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


class HiddenStateAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.embeddings = self.model.model.embed_tokens.weight.data.float().cpu().numpy()
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        print(f"  Ready")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return self.embeddings[ids[0]] if ids else None
    
    def _get_hidden(self, prompt: str) -> np.ndarray:
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
    
    def _decode(self, hidden: np.ndarray) -> str:
        logits = np.dot(self.lm_head, hidden)
        return self.tokenizer.decode([np.argmax(logits)]).strip()
    
    def analyze(self, pairs: List[Tuple[str, str]], template: str):
        """Analyze hidden state structure."""
        print(f"\nAnalyzing {len(pairs)} pairs...")
        
        # Collect data
        embeddings = []
        hiddens = []
        entities = []
        
        for entity, answer in pairs:
            emb = self._get_embedding(entity)
            hidden = self._get_hidden(template.format(entity=entity))
            
            embeddings.append(emb)
            hiddens.append(hidden)
            entities.append(entity)
        
        embeddings = np.array(embeddings)
        hiddens = np.array(hiddens)
        
        # 1. Linear regression: hidden = W @ embedding + b
        print("\n--- Linear Transform Analysis ---")
        reg = LinearRegression()
        reg.fit(embeddings, hiddens)
        
        predicted = reg.predict(embeddings)
        residuals = hiddens - predicted
        
        # Variance explained
        total_var = np.var(hiddens)
        residual_var = np.var(residuals)
        explained = 1 - residual_var / total_var
        print(f"  Variance explained by linear transform: {explained*100:.1f}%")
        
        # 2. Residual similarity
        print("\n--- Residual Analysis ---")
        mean_residual = np.mean(residuals, axis=0)
        
        # How similar are residuals to each other?
        sims = []
        for i in range(len(residuals)):
            for j in range(i+1, len(residuals)):
                sim = np.dot(residuals[i], residuals[j]) / (
                    np.linalg.norm(residuals[i]) * np.linalg.norm(residuals[j]) + 1e-10
                )
                sims.append(sim)
        
        print(f"  Mean residual similarity: {np.mean(sims):.3f}")
        print(f"  Residual similarity std: {np.std(sims):.3f}")
        
        # 3. Reconstruction test
        print("\n--- Reconstruction Test ---")
        print("  Method: hidden_reconstructed = W @ embedding + mean_residual")
        
        reconstructed = predicted + mean_residual
        
        correct = 0
        for i, entity in enumerate(entities):
            orig_token = self._decode(hiddens[i])
            recon_token = self._decode(reconstructed[i])
            
            match = orig_token == recon_token
            if match:
                correct += 1
            
            status = "✓" if match else "✗"
            print(f"    {entity}: orig='{orig_token}' recon='{recon_token}' {status}")
        
        accuracy = correct / len(entities)
        print(f"\n  Accuracy: {correct}/{len(entities)} = {accuracy*100:.1f}%")
        
        # 4. Test on NEW entities
        print("\n--- Generalization Test ---")
        test_pairs = [("Poland", "Warsaw"), ("Sweden", "Stockholm"), ("Norway", "Oslo")]
        
        test_correct = 0
        for entity, expected in test_pairs:
            emb = self._get_embedding(entity)
            
            # Predict hidden state
            pred_hidden = reg.predict([emb])[0] + mean_residual
            
            # Decode
            pred_token = self._decode(pred_hidden)
            
            # Compare to transformer
            actual_hidden = self._get_hidden(template.format(entity=entity))
            actual_token = self._decode(actual_hidden)
            
            match = pred_token == actual_token
            if match:
                test_correct += 1
            
            status = "✓" if match else "✗"
            print(f"    {entity}: pred='{pred_token}' actual='{actual_token}' {status}")
        
        test_accuracy = test_correct / len(test_pairs)
        print(f"\n  Generalization accuracy: {test_correct}/{len(test_pairs)} = {test_accuracy*100:.1f}%")
        
        return {
            'explained_variance': explained,
            'residual_similarity': np.mean(sims),
            'train_accuracy': accuracy,
            'test_accuracy': test_accuracy,
            'linear_transform': reg,
            'mean_residual': mean_residual,
        }


def main():
    print("=" * 70)
    print("HIDDEN STATE STRUCTURE ANALYSIS")
    print("=" * 70)
    
    analyzer = HiddenStateAnalyzer()
    
    pairs = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
        ("Japan", "Tokyo"),
        ("China", "Beijing"),
    ]
    
    results = analyzer.analyze(pairs, "The capital of {entity} is")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Variance explained by linear transform: {results['explained_variance']*100:.1f}%
Residual similarity: {results['residual_similarity']:.3f}
Training accuracy: {results['train_accuracy']*100:.1f}%
Generalization accuracy: {results['test_accuracy']*100:.1f}%

INTERPRETATION:
""")
    
    if results['test_accuracy'] > 0.5:
        print("SUCCESS: Hidden states CAN be decomposed!")
        print("  hidden = linear_transform(embedding) + shared_residual")
        print("  We can compute hidden states without the transformer!")
    else:
        print("FAILURE: Hidden states have entity-specific structure")
        print("  that cannot be predicted from embeddings alone.")
        print("  The transformer is doing something more complex.")


if __name__ == "__main__":
    main()
