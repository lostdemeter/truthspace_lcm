#!/usr/bin/env python3
"""
Hidden State Factorization: Shared vs Entity-Specific
======================================================

From φ-lattice analysis:
- φ-lattice reconstruction: 100% accuracy (k=16)
- Level correlation: 0.349 (some shared structure)
- Delta encoding: doesn't work (deltas too large)

New hypothesis: Hidden state = Shared_component + Entity_specific_component

If we can factor out the shared component:
1. Store shared component ONCE (for relationship type)
2. Store only entity-specific component per entity
3. This could be MUCH smaller

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

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)


class HiddenStateFactorizer:
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
    
    def factorize(self, pairs: List[Tuple[str, str]], template: str):
        """Factor hidden states into shared + entity-specific components."""
        print(f"\nCollecting {len(pairs)} hidden states...")
        
        hiddens = []
        entities = []
        
        for entity, answer in pairs:
            hidden = self._get_hidden(template.format(entity=entity))
            hiddens.append(hidden)
            entities.append(entity)
        
        hiddens = np.array(hiddens)
        
        # 1. Mean factorization: H = mean + residual
        print("\n--- Mean Factorization ---")
        mean_hidden = hiddens.mean(axis=0)
        residuals = hiddens - mean_hidden
        
        print(f"  Mean hidden norm: {np.linalg.norm(mean_hidden):.1f}")
        print(f"  Mean residual norm: {np.mean(np.linalg.norm(residuals, axis=1)):.1f}")
        print(f"  Residual/Mean ratio: {np.mean(np.linalg.norm(residuals, axis=1)) / np.linalg.norm(mean_hidden):.3f}")
        
        # Test reconstruction
        correct = 0
        for i in range(len(hiddens)):
            reconstructed = mean_hidden + residuals[i]
            _, orig_idx = self._decode(hiddens[i])
            _, recon_idx = self._decode(reconstructed)
            if orig_idx == recon_idx:
                correct += 1
        print(f"  Reconstruction accuracy: {correct}/{len(hiddens)} = {correct/len(hiddens)*100:.0f}%")
        
        # 2. PCA factorization: Find principal components
        print("\n--- PCA Factorization ---")
        
        # Center the data
        centered = hiddens - mean_hidden
        
        # PCA on residuals
        n_components = min(len(hiddens) - 1, 10)
        pca = PCA(n_components=n_components)
        projected = pca.fit_transform(centered)
        
        print(f"  Variance explained by {n_components} components: {pca.explained_variance_ratio_.sum()*100:.1f}%")
        
        # How many components needed for 90%, 95%, 99%?
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        for threshold in [0.9, 0.95, 0.99]:
            n_needed = np.searchsorted(cumsum, threshold) + 1
            print(f"  Components for {threshold*100:.0f}% variance: {n_needed}")
        
        # 3. Test: Store only top-k components per entity
        print("\n--- Compressed Storage Test ---")
        
        for k in [1, 2, 3, 5, 10]:
            if k > n_components:
                continue
            
            correct = 0
            for i in range(len(hiddens)):
                # Reconstruct from k components
                truncated = np.concatenate([projected[i, :k], np.zeros(n_components - k)])
                reconstructed = mean_hidden + pca.inverse_transform(truncated.reshape(1, -1))[0]
                
                _, orig_idx = self._decode(hiddens[i])
                _, recon_idx = self._decode(reconstructed)
                if orig_idx == recon_idx:
                    correct += 1
            
            # Storage: mean (shared) + k floats per entity
            storage_per_entity = k * 4  # float32
            accuracy = correct / len(hiddens)
            print(f"  k={k}: {accuracy*100:.0f}% accuracy, {storage_per_entity} bytes/entity")
        
        # 4. Analyze what the residual encodes
        print("\n--- Residual Analysis ---")
        
        # Does residual correlate with answer token?
        answer_ids = []
        for entity, answer in pairs:
            ids = self.tokenizer.encode(answer, add_special_tokens=False)
            answer_ids.append(ids[0] if ids else -1)
        answer_ids = np.array(answer_ids)
        
        # Correlation of each residual dimension with answer ID
        correlations = []
        for d in range(self.hidden_dim):
            if np.std(residuals[:, d]) > 1e-10:
                corr = np.corrcoef(residuals[:, d], answer_ids)[0, 1]
                correlations.append(abs(corr))
            else:
                correlations.append(0)
        
        correlations = np.array(correlations)
        top_dims = np.argsort(correlations)[-10:][::-1]
        
        print(f"  Top dimensions correlated with answer:")
        for d in top_dims:
            print(f"    Dim {d}: {correlations[d]:.3f}")
        
        # 5. Test: Can we predict residual from entity embedding?
        print("\n--- Residual Prediction from Entity Embedding ---")
        
        embeddings = self.model.model.embed_tokens.weight.data.float().cpu().numpy()
        
        entity_embs = []
        for entity, _ in pairs:
            ids = self.tokenizer.encode(entity, add_special_tokens=False)
            if ids:
                entity_embs.append(embeddings[ids[0]])
        entity_embs = np.array(entity_embs)
        
        # Linear regression: residual = W @ entity_embedding
        from sklearn.linear_model import Ridge
        
        reg = Ridge(alpha=1.0)
        reg.fit(entity_embs, residuals)
        
        predicted_residuals = reg.predict(entity_embs)
        
        # Test reconstruction
        correct = 0
        for i in range(len(hiddens)):
            reconstructed = mean_hidden + predicted_residuals[i]
            _, orig_idx = self._decode(hiddens[i])
            _, recon_idx = self._decode(reconstructed)
            if orig_idx == recon_idx:
                correct += 1
        
        print(f"  Accuracy with predicted residuals: {correct}/{len(hiddens)} = {correct/len(hiddens)*100:.0f}%")
        
        # 6. The key question: What's the MINIMAL entity-specific info?
        print("\n--- Minimal Entity-Specific Information ---")
        
        # Hypothesis: The answer token embedding IS the entity-specific info
        answer_embs = []
        for _, answer in pairs:
            ids = self.tokenizer.encode(answer, add_special_tokens=False)
            if ids:
                answer_embs.append(embeddings[ids[0]])
        answer_embs = np.array(answer_embs)
        
        # Does residual correlate with answer embedding?
        residual_answer_corr = []
        for i in range(len(residuals)):
            corr = np.dot(residuals[i], answer_embs[i]) / (
                np.linalg.norm(residuals[i]) * np.linalg.norm(answer_embs[i]) + 1e-10
            )
            residual_answer_corr.append(corr)
        
        print(f"  Mean residual-answer_embedding correlation: {np.mean(residual_answer_corr):.3f}")
        
        # Test: Can we reconstruct using mean + scaled answer embedding?
        print("\n--- Answer Embedding Reconstruction ---")
        
        for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            correct = 0
            for i in range(len(hiddens)):
                reconstructed = mean_hidden + scale * answer_embs[i]
                _, orig_idx = self._decode(hiddens[i])
                _, recon_idx = self._decode(reconstructed)
                if orig_idx == recon_idx:
                    correct += 1
            
            accuracy = correct / len(hiddens)
            print(f"  scale={scale}: {accuracy*100:.0f}% accuracy")
        
        return {
            'mean_hidden': mean_hidden,
            'residuals': residuals,
            'pca': pca,
            'projected': projected,
        }


def main():
    print("=" * 70)
    print("HIDDEN STATE FACTORIZATION")
    print("=" * 70)
    print("""
Goal: Factor hidden state into shared + entity-specific components.

If H = shared + entity_specific, we can:
1. Store 'shared' ONCE per relationship type
2. Store only 'entity_specific' per entity
3. Potentially MUCH smaller storage
""")
    
    factorizer = HiddenStateFactorizer()
    
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
    
    results = factorizer.factorize(pairs, "The capital of {entity} is")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    main()
