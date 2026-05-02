#!/usr/bin/env python3
"""
Hidden State Decomposition: What Makes Them Entity-Specific?
==============================================================

From DA2 (Doc 122, 125):
- DA2's 384 dimensions encode different features (depth, position, luminance)
- Depth is LINEARLY encoded in specific dimensions
- We could decode with a weighted sum: depth = Σ weight_i × dim_i
- Weights follow φ-patterns

Question: Does the LLM hidden state have similar structure?
- Are there dimensions that encode "entity type"?
- Are there dimensions that encode "answer"?
- Can we decompose: hidden = entity_component + relationship_component + answer_component?

If we can decompose the hidden state, we might be able to:
1. Store only the answer_component per entity
2. Compute entity_component from embedding
3. Combine them without the transformer

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2


class HiddenStateDecomposer:
    """
    Analyzes the structure of hidden states to understand what makes them entity-specific.
    """
    
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
        self.hidden_dim = self.model.config.hidden_size
        
        print(f"  Hidden dim: {self.hidden_dim}")
    
    def _get_entity_embedding(self, entity: str) -> np.ndarray:
        """Get embedding for entity."""
        ids = self.tokenizer.encode(entity, add_special_tokens=False)
        if not ids:
            return None
        return self.embeddings[ids[0]]
    
    def _get_hidden_state(self, prompt: str) -> np.ndarray:
        """Get final hidden state."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        first_device = next(self.model.parameters()).device
        input_ids = input_ids.to(first_device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
        
        return hidden
    
    def collect_data(self, pairs: List[Tuple[str, str]], template: str) -> Dict:
        """
        Collect hidden states and embeddings for analysis.
        """
        print(f"\nCollecting data for {len(pairs)} pairs...")
        
        data = {
            'entities': [],
            'answers': [],
            'entity_embeddings': [],
            'hidden_states': [],
            'answer_token_ids': [],
        }
        
        for entity, answer in pairs:
            prompt = template.format(entity=entity)
            
            # Get entity embedding
            emb = self._get_entity_embedding(entity)
            if emb is None:
                continue
            
            # Get hidden state
            hidden = self._get_hidden_state(prompt)
            
            # Get answer token ID
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            answer_id = answer_ids[0] if answer_ids else -1
            
            data['entities'].append(entity)
            data['answers'].append(answer)
            data['entity_embeddings'].append(emb)
            data['hidden_states'].append(hidden)
            data['answer_token_ids'].append(answer_id)
        
        # Convert to arrays
        data['entity_embeddings'] = np.array(data['entity_embeddings'])
        data['hidden_states'] = np.array(data['hidden_states'])
        data['answer_token_ids'] = np.array(data['answer_token_ids'])
        
        print(f"  Collected {len(data['entities'])} samples")
        return data
    
    def analyze_dimensions(self, data: Dict) -> Dict:
        """
        Analyze which dimensions encode what information.
        
        Like DA2, we look for dimensions that correlate with:
        - Entity embedding values
        - Answer token ID
        """
        print("\n--- Dimension Analysis ---")
        
        hidden_states = data['hidden_states']
        entity_embeddings = data['entity_embeddings']
        answer_ids = data['answer_token_ids']
        
        n_samples, hidden_dim = hidden_states.shape
        _, emb_dim = entity_embeddings.shape
        
        # 1. Which hidden dimensions correlate with entity embedding dimensions?
        entity_correlations = np.zeros((hidden_dim, emb_dim))
        for h in range(hidden_dim):
            for e in range(emb_dim):
                if np.std(hidden_states[:, h]) > 1e-10 and np.std(entity_embeddings[:, e]) > 1e-10:
                    entity_correlations[h, e] = np.corrcoef(hidden_states[:, h], entity_embeddings[:, e])[0, 1]
        
        # Max correlation per hidden dimension
        max_entity_corr = np.max(np.abs(entity_correlations), axis=1)
        
        # 2. Which hidden dimensions correlate with answer token ID?
        answer_correlations = np.zeros(hidden_dim)
        for h in range(hidden_dim):
            if np.std(hidden_states[:, h]) > 1e-10:
                answer_correlations[h] = np.corrcoef(hidden_states[:, h], answer_ids)[0, 1]
        
        # 3. Categorize dimensions
        entity_dims = np.where(max_entity_corr > 0.5)[0]
        answer_dims = np.where(np.abs(answer_correlations) > 0.3)[0]
        shared_dims = np.intersect1d(entity_dims, answer_dims)
        
        print(f"  Dimensions with entity correlation > 0.5: {len(entity_dims)}")
        print(f"  Dimensions with answer correlation > 0.3: {len(answer_dims)}")
        print(f"  Shared dimensions: {len(shared_dims)}")
        
        # Top entity-correlated dimensions
        top_entity = np.argsort(max_entity_corr)[-10:][::-1]
        print(f"\n  Top entity-correlated dimensions:")
        for d in top_entity:
            print(f"    Dim {d}: {max_entity_corr[d]:.3f}")
        
        # Top answer-correlated dimensions
        top_answer = np.argsort(np.abs(answer_correlations))[-10:][::-1]
        print(f"\n  Top answer-correlated dimensions:")
        for d in top_answer:
            print(f"    Dim {d}: {answer_correlations[d]:.3f}")
        
        return {
            'entity_correlations': entity_correlations,
            'max_entity_corr': max_entity_corr,
            'answer_correlations': answer_correlations,
            'entity_dims': entity_dims,
            'answer_dims': answer_dims,
        }
    
    def decompose_hidden_state(self, data: Dict) -> Dict:
        """
        Try to decompose hidden state into components.
        
        Hypothesis: hidden = f(entity_embedding) + answer_component
        
        If true, we could:
        1. Compute f(entity_embedding) for any entity
        2. Store answer_component per relationship type
        3. Combine without transformer
        """
        print("\n--- Hidden State Decomposition ---")
        
        hidden_states = data['hidden_states']
        entity_embeddings = data['entity_embeddings']
        
        # Try linear regression: hidden = W @ entity_embedding + b
        print("\n  Fitting: hidden = W @ entity_embedding + b")
        
        reg = LinearRegression()
        reg.fit(entity_embeddings, hidden_states)
        
        predicted_hidden = reg.predict(entity_embeddings)
        residuals = hidden_states - predicted_hidden
        
        # How much variance is explained?
        total_var = np.var(hidden_states, axis=0).sum()
        residual_var = np.var(residuals, axis=0).sum()
        explained_var = 1 - residual_var / total_var
        
        print(f"  Variance explained by linear transform: {explained_var*100:.1f}%")
        
        # Per-dimension analysis
        per_dim_explained = 1 - np.var(residuals, axis=0) / (np.var(hidden_states, axis=0) + 1e-10)
        well_explained = np.sum(per_dim_explained > 0.5)
        
        print(f"  Dimensions with >50% variance explained: {well_explained}/{self.hidden_dim}")
        
        # The residual is what's NOT explained by entity embedding
        # This might be the "relationship + answer" component
        print(f"\n  Residual analysis:")
        print(f"    Mean residual norm: {np.mean(np.linalg.norm(residuals, axis=1)):.2f}")
        print(f"    Std residual norm: {np.std(np.linalg.norm(residuals, axis=1)):.2f}")
        
        # Are residuals similar across entities? (Would indicate shared relationship component)
        residual_similarities = []
        for i in range(len(residuals)):
            for j in range(i+1, len(residuals)):
                sim = np.dot(residuals[i], residuals[j]) / (
                    np.linalg.norm(residuals[i]) * np.linalg.norm(residuals[j]) + 1e-10
                )
                residual_similarities.append(sim)
        
        print(f"    Mean residual similarity: {np.mean(residual_similarities):.3f}")
        print(f"    Std residual similarity: {np.std(residual_similarities):.3f}")
        
        return {
            'linear_transform': reg,
            'predicted_hidden': predicted_hidden,
            'residuals': residuals,
            'explained_variance': explained_var,
            'per_dim_explained': per_dim_explained,
            'residual_similarities': np.array(residual_similarities),
        }
    
    def test_reconstruction(self, data: Dict, decomposition: Dict) -> Dict:
        """
        Test if we can reconstruct hidden states and decode correctly.
        """
        print("\n--- Reconstruction Test ---")
        
        hidden_states = data['hidden_states']
        entity_embeddings = data['entity_embeddings']
        
        # Method 1: Use linear transform + mean residual
        reg = decomposition['linear_transform']
        mean_residual = np.mean(decomposition['residuals'], axis=0)
        
        reconstructed = reg.predict(entity_embeddings) + mean_residual
        
        # Decode from reconstructed
        correct = 0
        for i, entity in enumerate(data['entities']):
            # Original decode
            orig_logits = np.dot(self.lm_head, hidden_states[i])
            orig_token = self.tokenizer.decode([np.argmax(orig_logits)])
            
            # Reconstructed decode
            recon_logits = np.dot(self.lm_head, reconstructed[i])
            recon_token = self.tokenizer.decode([np.argmax(recon_logits)])
            
            match = orig_token == recon_token
            if match:
                correct += 1
            
            status = "✓" if match else "✗"
            print(f"  {entity}: orig='{orig_token.strip()}' recon='{recon_token.strip()}' {status}")
        
        accuracy = correct / len(data['entities'])
        print(f"\n  Reconstruction accuracy: {correct}/{len(data['entities'])} = {accuracy*100:.1f}%")
        
        return {
            'accuracy': accuracy,
            'reconstructed': reconstructed,
        }


def main():
    print("=" * 70)
    print("HIDDEN STATE DECOMPOSITION")
    print("=" * 70)
    print("""
Goal: Understand what makes hidden states entity-specific.

From DA2: The backbone encodes features in specific dimensions.
Question: Does the LLM hidden state have similar structure?

If hidden = f(entity_embedding) + relationship_component + answer_component,
we might be able to compute it without the transformer.
""")
    
    decomposer = HiddenStateDecomposer()
    
    # Collect data
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
    ]
    
    data = decomposer.collect_data(pairs, "The capital of {entity} is")
    
    # Analyze dimensions
    dim_analysis = decomposer.analyze_dimensions(data)
    
    # Decompose hidden state
    decomposition = decomposer.decompose_hidden_state(data)
    
    # Test reconstruction
    reconstruction = decomposer.test_reconstruction(data, decomposition)
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    
    explained = decomposition['explained_variance']
    accuracy = reconstruction['accuracy']
    residual_sim = np.mean(decomposition['residual_similarities'])
    
    print(f"""
Key Findings:

1. LINEAR TRANSFORM FROM ENTITY EMBEDDING
   - Explains {explained*100:.1f}% of hidden state variance
   - {'GOOD' if explained > 0.5 else 'POOR'}: {'Most' if explained > 0.5 else 'Little'} of hidden state is predictable from entity

2. RESIDUAL STRUCTURE
   - Mean residual similarity: {residual_sim:.3f}
   - {'GOOD' if residual_sim > 0.5 else 'POOR'}: Residuals are {'similar' if residual_sim > 0.5 else 'different'} across entities

3. RECONSTRUCTION ACCURACY
   - {accuracy*100:.1f}% of tokens decoded correctly
   - {'SUCCESS' if accuracy > 0.8 else 'FAILURE'}: {'Can' if accuracy > 0.8 else 'Cannot'} reconstruct without transformer

IMPLICATION:
""")
    
    if accuracy > 0.8:
        print("We CAN decompose hidden states and reconstruct without transformer!")
        print("The entity-specific part is predictable from the embedding.")
    else:
        print("Hidden states have entity-specific structure that's NOT predictable")
        print("from embeddings alone. The transformer is doing something more complex.")


if __name__ == "__main__":
    main()
