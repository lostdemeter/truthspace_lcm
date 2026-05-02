#!/usr/bin/env python3
"""
Hidden State Transition: Can We Predict the Next Hidden State?
===============================================================

The key question: Given hidden_state_N and token_N, can we predict hidden_state_{N+1}
WITHOUT running the transformer?

If yes, we can fully replace the transformer with:
1. Store initial hidden state for prompt
2. Decode token
3. Predict next hidden state (no transformer!)
4. Repeat

From Doc 183: Navigation is 99.58% universal within relationship types.
This suggests the transition function might be learnable.

Author: TruthSpace LCM Team
Date: 2026-02-01
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2


class HiddenStateAnalyzer:
    """Analyze hidden state transitions during generation."""
    
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
        
        # Get embedding matrix
        self.embeddings = self.model.model.embed_tokens.weight.data.float().cpu()
        
        print(f"  Layers: {self.n_layers}, Hidden dim: {self.hidden_dim}")
    
    def get_generation_trajectory(self, prompt: str, n_tokens: int = 5) -> Dict:
        """
        Generate tokens and capture hidden states at each step.
        
        Returns dict with:
        - tokens: list of generated tokens
        - hidden_states: list of final hidden states at each step
        - token_embeddings: embeddings of each generated token
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        first_device = next(self.model.parameters()).device
        input_ids = input_ids.to(first_device)
        
        tokens = []
        hidden_states = []
        token_embeddings = []
        
        current_ids = input_ids
        
        for i in range(n_tokens):
            with torch.no_grad():
                outputs = self.model(current_ids, output_hidden_states=True)
                
                # Get final hidden state (last layer, last position)
                final_hidden = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
                hidden_states.append(final_hidden)
                
                # Get next token
                logits = outputs.logits[0, -1, :]
                next_token_id = logits.argmax().item()
                next_token = self.tokenizer.decode([next_token_id])
                tokens.append(next_token)
                
                # Get embedding of the token we're about to add
                token_emb = self.embeddings[next_token_id].numpy()
                token_embeddings.append(token_emb)
                
                # Append token for next iteration
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([[next_token_id]], device=first_device)
                ], dim=1)
        
        return {
            'prompt': prompt,
            'tokens': tokens,
            'hidden_states': np.array(hidden_states),
            'token_embeddings': np.array(token_embeddings),
        }
    
    def analyze_transitions(self, trajectory: Dict) -> Dict:
        """
        Analyze the transitions between hidden states.
        
        Key question: Is there a pattern we can learn?
        """
        hidden_states = trajectory['hidden_states']
        token_embeddings = trajectory['token_embeddings']
        
        n_steps = len(hidden_states) - 1
        
        # Compute deltas between consecutive hidden states
        deltas = []
        for i in range(n_steps):
            delta = hidden_states[i + 1] - hidden_states[i]
            deltas.append(delta)
        
        deltas = np.array(deltas)
        
        # Analyze delta properties
        delta_norms = np.linalg.norm(deltas, axis=1)
        hidden_norms = np.linalg.norm(hidden_states[:-1], axis=1)
        
        # Check if delta is related to token embedding
        embedding_correlations = []
        for i in range(n_steps):
            # Correlation between delta and token embedding
            corr = np.corrcoef(deltas[i], token_embeddings[i])[0, 1]
            embedding_correlations.append(corr)
        
        # Check if deltas are similar to each other (universal transition?)
        delta_similarities = []
        for i in range(n_steps):
            for j in range(i + 1, n_steps):
                sim = np.dot(deltas[i], deltas[j]) / (
                    np.linalg.norm(deltas[i]) * np.linalg.norm(deltas[j]) + 1e-10
                )
                delta_similarities.append(sim)
        
        return {
            'n_steps': n_steps,
            'delta_norms': delta_norms,
            'hidden_norms': hidden_norms,
            'relative_change': delta_norms / hidden_norms,
            'embedding_correlations': np.array(embedding_correlations),
            'delta_similarities': np.array(delta_similarities) if delta_similarities else np.array([0]),
            'deltas': deltas,
        }
    
    def test_linear_prediction(self, trajectory: Dict) -> Dict:
        """
        Test if we can predict next hidden state with a linear model:
        
        h_{n+1} = h_n + W @ token_embedding_n + b
        
        This is the simplest possible transition function.
        """
        hidden_states = trajectory['hidden_states']
        token_embeddings = trajectory['token_embeddings']
        
        n_steps = len(hidden_states) - 1
        if n_steps < 2:
            return {'error': 'Need at least 2 transitions to test'}
        
        # Compute targets (deltas)
        deltas = hidden_states[1:] - hidden_states[:-1]
        
        # Simple prediction: delta = alpha * token_embedding
        # Find best alpha for each dimension
        predictions = []
        for i in range(n_steps):
            # Predict delta as scaled token embedding
            # This assumes delta ∝ token_embedding
            
            # Find scaling factor
            scale = np.dot(deltas[i], token_embeddings[i]) / (
                np.dot(token_embeddings[i], token_embeddings[i]) + 1e-10
            )
            
            pred_delta = scale * token_embeddings[i]
            predictions.append(pred_delta)
        
        predictions = np.array(predictions)
        
        # Compute prediction error
        errors = deltas - predictions
        error_norms = np.linalg.norm(errors, axis=1)
        delta_norms = np.linalg.norm(deltas, axis=1)
        
        relative_errors = error_norms / (delta_norms + 1e-10)
        
        # Predict hidden states
        predicted_hidden = []
        for i in range(n_steps):
            pred_h = hidden_states[i] + predictions[i]
            predicted_hidden.append(pred_h)
        
        predicted_hidden = np.array(predicted_hidden)
        actual_hidden = hidden_states[1:]
        
        # Correlation between predicted and actual
        correlations = []
        for i in range(n_steps):
            corr = np.corrcoef(predicted_hidden[i], actual_hidden[i])[0, 1]
            correlations.append(corr)
        
        return {
            'relative_errors': relative_errors,
            'mean_relative_error': np.mean(relative_errors),
            'correlations': np.array(correlations),
            'mean_correlation': np.mean(correlations),
        }
    
    def test_additive_prediction(self, trajectory: Dict) -> Dict:
        """
        Test if hidden state change is simply additive:
        
        h_{n+1} = h_n + constant_delta
        
        This would mean the transition is independent of the token!
        """
        hidden_states = trajectory['hidden_states']
        
        n_steps = len(hidden_states) - 1
        if n_steps < 2:
            return {'error': 'Need at least 2 transitions'}
        
        # Compute deltas
        deltas = hidden_states[1:] - hidden_states[:-1]
        
        # Use mean delta as the constant
        mean_delta = np.mean(deltas, axis=0)
        
        # Predict using constant delta
        predicted_hidden = []
        for i in range(n_steps):
            pred_h = hidden_states[i] + mean_delta
            predicted_hidden.append(pred_h)
        
        predicted_hidden = np.array(predicted_hidden)
        actual_hidden = hidden_states[1:]
        
        # Compute errors
        errors = actual_hidden - predicted_hidden
        error_norms = np.linalg.norm(errors, axis=1)
        actual_norms = np.linalg.norm(actual_hidden, axis=1)
        
        relative_errors = error_norms / actual_norms
        
        # Correlations
        correlations = []
        for i in range(n_steps):
            corr = np.corrcoef(predicted_hidden[i], actual_hidden[i])[0, 1]
            correlations.append(corr)
        
        return {
            'mean_delta_norm': np.linalg.norm(mean_delta),
            'relative_errors': relative_errors,
            'mean_relative_error': np.mean(relative_errors),
            'correlations': np.array(correlations),
            'mean_correlation': np.mean(correlations),
        }


def main():
    print("=" * 70)
    print("HIDDEN STATE TRANSITION ANALYSIS")
    print("=" * 70)
    print("""
Question: Can we predict the next hidden state without the transformer?

If h_{n+1} = f(h_n, token_n) is learnable, we can:
1. Store initial hidden state
2. Decode token
3. Predict next hidden state (no transformer!)
4. Repeat
""")
    
    analyzer = HiddenStateAnalyzer()
    
    # Test with capital query
    print("\n" + "=" * 50)
    print("TEST 1: Capital Query")
    print("=" * 50)
    
    trajectory = analyzer.get_generation_trajectory(
        "The capital of France is",
        n_tokens=5
    )
    
    print(f"\nPrompt: '{trajectory['prompt']}'")
    print(f"Generated: {''.join(trajectory['tokens'])}")
    
    analysis = analyzer.analyze_transitions(trajectory)
    
    print(f"\n--- Transition Analysis ---")
    print(f"Steps: {analysis['n_steps']}")
    print(f"Delta norms: {analysis['delta_norms']}")
    print(f"Relative change: {analysis['relative_change']}")
    print(f"Embedding correlations: {analysis['embedding_correlations']}")
    print(f"Delta similarities: mean={np.mean(analysis['delta_similarities']):.4f}")
    
    # Test linear prediction
    print(f"\n--- Linear Prediction Test ---")
    linear = analyzer.test_linear_prediction(trajectory)
    print(f"Mean relative error: {linear['mean_relative_error']:.4f}")
    print(f"Mean correlation: {linear['mean_correlation']:.4f}")
    
    # Test additive prediction
    print(f"\n--- Additive Prediction Test ---")
    additive = analyzer.test_additive_prediction(trajectory)
    print(f"Mean relative error: {additive['mean_relative_error']:.4f}")
    print(f"Mean correlation: {additive['mean_correlation']:.4f}")
    
    # Test with general query
    print("\n" + "=" * 50)
    print("TEST 2: General Query")
    print("=" * 50)
    
    trajectory2 = analyzer.get_generation_trajectory(
        "Hello, how are you",
        n_tokens=10
    )
    
    print(f"\nPrompt: '{trajectory2['prompt']}'")
    print(f"Generated: {''.join(trajectory2['tokens'])}")
    
    analysis2 = analyzer.analyze_transitions(trajectory2)
    linear2 = analyzer.test_linear_prediction(trajectory2)
    additive2 = analyzer.test_additive_prediction(trajectory2)
    
    print(f"\n--- Results ---")
    print(f"Linear prediction correlation: {linear2['mean_correlation']:.4f}")
    print(f"Additive prediction correlation: {additive2['mean_correlation']:.4f}")
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    
    if linear['mean_correlation'] > 0.99 or additive['mean_correlation'] > 0.99:
        print("""
GOOD NEWS: Hidden state transitions are highly predictable!

We can potentially replace the transformer with:
1. Store initial hidden state for prompt
2. Apply learned transition function
3. Decode token
4. Repeat

This would give us full generation without any transformer layers!
""")
    else:
        print(f"""
FINDINGS:
- Linear prediction correlation: {linear['mean_correlation']:.4f}
- Additive prediction correlation: {additive['mean_correlation']:.4f}

The simple models don't capture the full transition.
The transformer layers are doing something more complex.

Options:
1. Learn a more complex transition function (neural net)
2. Store full response trajectories (precache)
3. Use the 99.58% universal navigation pattern differently
""")


if __name__ == "__main__":
    main()
