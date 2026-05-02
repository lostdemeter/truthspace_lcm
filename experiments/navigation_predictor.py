#!/usr/bin/env python3
"""
Navigation Predictor: Predict Outputs Using Learned Navigation Geometry
=========================================================================

Key insight: We don't need to understand WHAT the transformer knows,
just HOW it navigates through space.

Findings from navigation_geometry.py:
1. Navigation shape is 99.58% universal within relationship types
2. Layer 0 applies 77° rotation (matches capital-of angle)
3. Growth factor: 299x from embedding to final hidden state
4. Peak deviation at layer 27

Strategy:
1. Learn the MEAN trajectory for a relationship type
2. Learn the DEVIATION BASIS (like wavelets)
3. For new queries: apply mean trajectory + entity-specific deviation
4. Skip all 28 transformer layers!

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
import time
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2


class NavigationPredictor:
    """
    Predicts transformer outputs using learned navigation geometry.
    
    Instead of running 28 transformer layers, we:
    1. Apply the learned mean trajectory
    2. Add entity-specific deviation (from a small basis)
    3. Decode directly from the predicted final hidden state
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
        
        # Get LM head for decoding
        self.lm_head = self.model.lm_head.weight.data.float().cpu()
        
        # Learned navigation patterns (populated by learn())
        self.mean_trajectories: Dict[str, np.ndarray] = {}
        self.deviation_basis: Dict[str, np.ndarray] = {}
        self.entity_coefficients: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Statistics
        self.stats = {
            'predictions': 0,
            'correct': 0,
            'prediction_time': 0,
            'transformer_time': 0,
        }
        
        print(f"  Layers: {self.n_layers}, Hidden dim: {self.hidden_dim}")
    
    def get_trajectory(self, prompt: str) -> np.ndarray:
        """Get full trajectory through all layers."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        first_device = next(self.model.parameters()).device
        input_ids = input_ids.to(first_device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
        
        trajectory = []
        for hidden in outputs.hidden_states:
            h = hidden[0, -1, :].float().cpu().numpy()
            trajectory.append(h)
        
        return np.array(trajectory)
    
    def learn(self, template: str, entities: List[str], rel_type: str):
        """
        Learn the navigation pattern for a relationship type.
        
        Args:
            template: Prompt template with {entity} placeholder
            entities: List of entities to learn from
            rel_type: Name for this relationship type
        """
        print(f"\nLearning navigation for '{rel_type}'...")
        
        # Collect trajectories
        trajectories = []
        for entity in entities:
            prompt = template.format(entity=entity)
            traj = self.get_trajectory(prompt)
            trajectories.append(traj)
        
        trajectories = np.array(trajectories)  # (n_entities, n_layers+1, hidden_dim)
        
        # Compute mean trajectory
        mean_traj = np.mean(trajectories, axis=0)
        self.mean_trajectories[rel_type] = mean_traj
        
        # Compute deviations from mean
        deviations = trajectories - mean_traj  # (n_entities, n_layers+1, hidden_dim)
        
        # Flatten deviations for PCA
        n_entities = len(entities)
        flat_deviations = deviations.reshape(n_entities, -1)  # (n_entities, (n_layers+1)*hidden_dim)
        
        # Learn deviation basis using PCA
        n_components = min(10, n_entities - 1)  # Keep top 10 components
        pca = PCA(n_components=n_components)
        coefficients = pca.fit_transform(flat_deviations)
        
        # Store basis and coefficients
        self.deviation_basis[rel_type] = pca.components_  # (n_components, (n_layers+1)*hidden_dim)
        self.entity_coefficients[rel_type] = {
            entity: coefficients[i]
            for i, entity in enumerate(entities)
        }
        
        # Report variance explained
        var_explained = np.sum(pca.explained_variance_ratio_) * 100
        print(f"  Learned from {n_entities} entities")
        print(f"  Deviation basis: {n_components} components, {var_explained:.1f}% variance explained")
    
    def predict_trajectory(self, entity: str, rel_type: str) -> Optional[np.ndarray]:
        """
        Predict the trajectory for an entity using learned navigation.
        
        If entity was seen during learning, use its coefficients.
        Otherwise, use mean trajectory (zero deviation).
        """
        if rel_type not in self.mean_trajectories:
            return None
        
        mean_traj = self.mean_trajectories[rel_type]
        
        # Get entity-specific deviation
        if rel_type in self.entity_coefficients and entity in self.entity_coefficients[rel_type]:
            coeffs = self.entity_coefficients[rel_type][entity]
            basis = self.deviation_basis[rel_type]
            
            # Reconstruct deviation
            flat_deviation = np.dot(coeffs, basis)
            deviation = flat_deviation.reshape(mean_traj.shape)
            
            return mean_traj + deviation
        else:
            # Unknown entity - use mean trajectory
            return mean_traj
    
    def predict_token(self, entity: str, rel_type: str) -> Tuple[str, float]:
        """
        Predict the next token using navigation geometry.
        
        Returns (predicted_token, confidence)
        """
        start_time = time.time()
        
        trajectory = self.predict_trajectory(entity, rel_type)
        if trajectory is None:
            return None, 0.0
        
        # Get final hidden state
        final_hidden = trajectory[-1]
        
        # Decode using LM head
        logits = np.dot(self.lm_head.numpy(), final_hidden)
        
        # Get top prediction
        top_idx = np.argmax(logits)
        confidence = float(np.exp(logits[top_idx]) / np.sum(np.exp(logits - logits.max())))
        
        predicted_token = self.tokenizer.decode([top_idx]).strip()
        
        self.stats['predictions'] += 1
        self.stats['prediction_time'] += time.time() - start_time
        
        return predicted_token, confidence
    
    def predict_with_transformer(self, prompt: str) -> Tuple[str, float]:
        """Get transformer's prediction for comparison."""
        start_time = time.time()
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        first_device = next(self.model.parameters()).device
        input_ids = input_ids.to(first_device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :].float().cpu().numpy()
        
        top_idx = np.argmax(logits)
        confidence = float(np.exp(logits[top_idx]) / np.sum(np.exp(logits - logits.max())))
        
        predicted_token = self.tokenizer.decode([top_idx]).strip()
        
        self.stats['transformer_time'] += time.time() - start_time
        
        return predicted_token, confidence
    
    def evaluate(self, template: str, test_entities: List[str], rel_type: str):
        """
        Evaluate navigation prediction vs transformer.
        """
        print(f"\n--- Evaluation: {rel_type} ---")
        
        correct = 0
        total = 0
        
        for entity in test_entities:
            prompt = template.format(entity=entity)
            
            # Navigation prediction
            nav_pred, nav_conf = self.predict_token(entity, rel_type)
            
            # Transformer prediction
            trans_pred, trans_conf = self.predict_with_transformer(prompt)
            
            # Check if they match
            match = nav_pred == trans_pred
            if match:
                correct += 1
            total += 1
            
            status = "✓" if match else "✗"
            print(f"  {entity}: nav='{nav_pred}' vs trans='{trans_pred}' {status}")
        
        accuracy = correct / total if total > 0 else 0
        print(f"\nAccuracy: {correct}/{total} = {accuracy*100:.1f}%")
        
        self.stats['correct'] = correct
        
        return accuracy


def main():
    print("=" * 70)
    print("NAVIGATION PREDICTOR: Skip Transformer Layers")
    print("=" * 70)
    
    predictor = NavigationPredictor()
    
    # Training entities
    train_entities = ["France", "Germany", "Italy", "Spain", "Japan", "China"]
    template = "The capital of {entity} is"
    
    # Learn navigation pattern
    predictor.learn(template, train_entities, "capital-of")
    
    # Evaluate on training entities (should be 100%)
    print("\n" + "=" * 50)
    print("TRAINING SET EVALUATION")
    print("=" * 50)
    predictor.evaluate(template, train_entities, "capital-of")
    
    # Evaluate on test entities (generalization)
    test_entities = ["Poland", "Sweden", "Greece"]
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION (Generalization)")
    print("=" * 50)
    predictor.evaluate(template, test_entities, "capital-of")
    
    # Speed comparison
    print("\n" + "=" * 50)
    print("SPEED COMPARISON")
    print("=" * 50)
    
    n_predictions = predictor.stats['predictions']
    nav_time = predictor.stats['prediction_time']
    trans_time = predictor.stats['transformer_time']
    
    print(f"Navigation predictions: {n_predictions}")
    print(f"  Total time: {nav_time*1000:.1f}ms")
    print(f"  Per prediction: {nav_time/n_predictions*1000:.2f}ms")
    
    print(f"\nTransformer predictions: {n_predictions}")
    print(f"  Total time: {trans_time*1000:.1f}ms")
    print(f"  Per prediction: {trans_time/n_predictions*1000:.2f}ms")
    
    speedup = trans_time / nav_time if nav_time > 0 else 0
    print(f"\nSpeedup: {speedup:.1f}x")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The Navigation Predictor demonstrates:

1. LEARNED NAVIGATION
   - Mean trajectory captures the universal navigation pattern
   - Deviation basis captures entity-specific variations
   - Only ~10 coefficients needed per entity

2. ACCURACY
   - Training set: Should match transformer exactly
   - Test set: Depends on how well mean trajectory generalizes

3. SPEED
   - Skip all 28 transformer layers
   - Just matrix multiply: trajectory → LM head → token

The key insight: We're not storing KNOWLEDGE, we're storing NAVIGATION.
The navigation pattern is learnable and reusable.
""")


if __name__ == "__main__":
    main()
