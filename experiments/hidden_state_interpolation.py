#!/usr/bin/env python3
"""
Hidden State Interpolation: Generalize Without Transformer
============================================================

Key insight: We can't COMPUTE hidden states geometrically, but we can
INTERPOLATE between known hidden states.

If we have:
- France → hidden_france → "Paris"
- Germany → hidden_germany → "Berlin"

Can we interpolate to get:
- Poland → interpolated_hidden → "Warsaw"?

The hypothesis: Hidden states for similar entities are geometrically close.
Interpolation in hidden space might give us the right answer.

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
import time
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2


class HiddenStateInterpolator:
    """
    Interpolates hidden states to handle new entities.
    
    Strategy:
    1. Store (entity_embedding, hidden_state, answer) for known entities
    2. For new entity: find nearest neighbors in embedding space
    3. Interpolate their hidden states
    4. Decode from interpolated hidden state
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Embeddings and LM head
        self.embeddings = self.model.model.embed_tokens.weight.data.float().cpu().numpy()
        self.lm_head_gpu = self.model.lm_head.weight.data.float()
        
        # Learned data per relationship type
        self.known_entities: Dict[str, List[str]] = {}  # rel_type → [entities]
        self.entity_embeddings: Dict[str, Dict[str, np.ndarray]] = {}  # rel_type → {entity: embedding}
        self.hidden_states: Dict[str, Dict[str, np.ndarray]] = {}  # rel_type → {entity: hidden}
        self.answers: Dict[str, Dict[str, str]] = {}  # rel_type → {entity: answer}
        
        # Nearest neighbor models
        self.nn_models: Dict[str, NearestNeighbors] = {}
        
        print(f"  Ready")
    
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
    
    def _decode_token(self, hidden: np.ndarray) -> Tuple[str, float]:
        """Decode token from hidden state."""
        hidden_gpu = torch.tensor(hidden, device=self.lm_head_gpu.device, dtype=torch.float32)
        logits = torch.matmul(self.lm_head_gpu, hidden_gpu)
        top_idx = logits.argmax().item()
        
        probs = torch.softmax(logits - logits.max(), dim=0)
        confidence = probs[top_idx].item()
        
        token = self.tokenizer.decode([top_idx])
        return token.strip(), confidence
    
    def learn(self, pairs: List[Tuple[str, str]], rel_type: str, template: str):
        """
        Learn hidden states for known entity pairs.
        """
        print(f"\nLearning: {rel_type}")
        
        self.known_entities[rel_type] = []
        self.entity_embeddings[rel_type] = {}
        self.hidden_states[rel_type] = {}
        self.answers[rel_type] = {}
        
        embeddings_list = []
        
        for entity, answer in pairs:
            prompt = template.format(entity=entity)
            
            # Get entity embedding
            emb = self._get_entity_embedding(entity)
            if emb is None:
                continue
            
            # Get hidden state
            hidden = self._get_hidden_state(prompt)
            
            # Store
            self.known_entities[rel_type].append(entity)
            self.entity_embeddings[rel_type][entity] = emb
            self.hidden_states[rel_type][entity] = hidden
            self.answers[rel_type][entity] = answer
            embeddings_list.append(emb)
        
        # Build nearest neighbor model
        embeddings_array = np.array(embeddings_list)
        self.nn_models[rel_type] = NearestNeighbors(n_neighbors=min(3, len(pairs)), metric='cosine')
        self.nn_models[rel_type].fit(embeddings_array)
        
        print(f"  Learned {len(pairs)} entities")
    
    def generate(self, entity: str, rel_type: str, k: int = 3) -> Tuple[str, float, str]:
        """
        Generate answer by interpolating hidden states.
        
        For known entities: use stored hidden state (exact)
        For unknown entities: interpolate from nearest neighbors
        """
        if rel_type not in self.known_entities:
            return None, 0.0, "unknown_relationship"
        
        # Check if entity is known
        if entity in self.hidden_states[rel_type]:
            hidden = self.hidden_states[rel_type][entity]
            token, conf = self._decode_token(hidden)
            return token, conf, "exact"
        
        # Unknown entity - interpolate
        emb = self._get_entity_embedding(entity)
        if emb is None:
            return None, 0.0, "no_embedding"
        
        # Find nearest neighbors
        k = min(k, len(self.known_entities[rel_type]))
        distances, indices = self.nn_models[rel_type].kneighbors([emb], n_neighbors=k)
        
        # Interpolate hidden states (weighted by inverse distance)
        weights = 1.0 / (distances[0] + 1e-10)
        weights = weights / weights.sum()
        
        interpolated_hidden = np.zeros_like(list(self.hidden_states[rel_type].values())[0])
        
        for i, idx in enumerate(indices[0]):
            neighbor_entity = self.known_entities[rel_type][idx]
            neighbor_hidden = self.hidden_states[rel_type][neighbor_entity]
            interpolated_hidden += weights[i] * neighbor_hidden
        
        # Decode
        token, conf = self._decode_token(interpolated_hidden)
        
        return token, conf, "interpolated"
    
    def generate_with_transformer(self, prompt: str) -> Tuple[str, float]:
        """Generate using transformer for comparison."""
        hidden = self._get_hidden_state(prompt)
        token, conf = self._decode_token(hidden)
        return token, conf


def main():
    print("=" * 70)
    print("HIDDEN STATE INTERPOLATION")
    print("=" * 70)
    print("""
Strategy: Store hidden states for known entities, interpolate for new ones.

If France→Paris and Germany→Berlin are geometrically close in hidden space,
then Poland (nearby in embedding space) should interpolate to Warsaw.
""")
    
    interpolator = HiddenStateInterpolator()
    
    # Learn from training pairs
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
    ]
    
    interpolator.learn(train_pairs, "capital-of", "The capital of {entity} is")
    
    # Test on training data (should be 100% - exact lookup)
    print("\n" + "=" * 50)
    print("TEST 1: Training Data (exact lookup)")
    print("=" * 50)
    
    correct = 0
    for entity, expected in train_pairs:
        answer, conf, method = interpolator.generate(entity, "capital-of")
        trans_answer, _ = interpolator.generate_with_transformer(f"The capital of {entity} is")
        
        match = answer == trans_answer
        if match:
            correct += 1
        
        status = "✓" if match else "✗"
        print(f"  {entity}: '{answer}' vs trans='{trans_answer}' [{method}] {status}")
    
    print(f"\nAccuracy: {correct}/{len(train_pairs)} = {correct/len(train_pairs)*100:.1f}%")
    
    # Test on new entities (interpolation)
    print("\n" + "=" * 50)
    print("TEST 2: New Entities (interpolation)")
    print("=" * 50)
    
    test_pairs = [
        ("Poland", "Warsaw"),
        ("Sweden", "Stockholm"),
        ("Norway", "Oslo"),
        ("Austria", "Vienna"),
        ("Greece", "Athens"),
        ("Portugal", "Lisbon"),
    ]
    
    correct = 0
    for entity, expected in test_pairs:
        answer, conf, method = interpolator.generate(entity, "capital-of")
        trans_answer, _ = interpolator.generate_with_transformer(f"The capital of {entity} is")
        
        match = answer == trans_answer
        if match:
            correct += 1
        
        status = "✓" if match else "✗"
        print(f"  {entity}: '{answer}' vs trans='{trans_answer}' [{method}] {status}")
    
    print(f"\nAccuracy: {correct}/{len(test_pairs)} = {correct/len(test_pairs)*100:.1f}%")
    
    # Speed benchmark
    print("\n" + "=" * 50)
    print("SPEED BENCHMARK")
    print("=" * 50)
    
    # Exact lookup (training data)
    n_iter = 100
    start = time.time()
    for _ in range(n_iter):
        for entity, _ in train_pairs:
            interpolator.generate(entity, "capital-of")
    exact_time = time.time() - start
    exact_per = exact_time / (n_iter * len(train_pairs)) * 1000
    
    # Interpolation (test data)
    start = time.time()
    for _ in range(n_iter):
        for entity, _ in test_pairs:
            interpolator.generate(entity, "capital-of")
    interp_time = time.time() - start
    interp_per = interp_time / (n_iter * len(test_pairs)) * 1000
    
    # Transformer
    start = time.time()
    for entity, _ in train_pairs[:3]:
        interpolator.generate_with_transformer(f"The capital of {entity} is")
    trans_time = time.time() - start
    trans_per = trans_time / 3 * 1000
    
    print(f"\nExact lookup: {exact_per:.2f} ms/query")
    print(f"Interpolation: {interp_per:.2f} ms/query")
    print(f"Transformer: {trans_per:.2f} ms/query")
    print(f"\nSpeedup (exact): {trans_per / exact_per:.1f}x")
    print(f"Speedup (interp): {trans_per / interp_per:.1f}x")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
