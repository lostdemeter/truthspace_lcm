#!/usr/bin/env python3
"""
Early Exit Test: Use Hidden States from Earlier Layers
=======================================================

Test if we can predict tokens using hidden states from earlier layers.
This avoids the complexity of manually calling layers.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import List, Tuple
import time

PHI = (1 + 5**0.5) / 2


class EarlyExitTransformer:
    """Test early exit by using hidden states from intermediate layers."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Loading model...")
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.device = next(self.model.parameters()).device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get final norm and lm_head for early exit
        self.final_norm = self.model.model.norm
        self.lm_head = self.model.lm_head
        
        print(f"Loaded on {self.device}")
    
    def forward_full(self, token_ids: List[int]) -> Tuple[int, float]:
        """Full forward pass."""
        ids = torch.tensor([token_ids]).to(self.device)
        
        with torch.no_grad():
            out = self.model(ids)
            logits = out.logits[0, -1]
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max().item()
            pred = torch.argmax(logits).item()
        
        return pred, confidence
    
    def forward_early_exit(self, token_ids: List[int], 
                           exit_layer: int) -> Tuple[int, float]:
        """
        Early exit: use hidden state from specified layer.
        
        Run full forward but use intermediate hidden state for prediction.
        """
        ids = torch.tensor([token_ids]).to(self.device)
        
        with torch.no_grad():
            # Get all hidden states
            out = self.model(ids, output_hidden_states=True)
            
            # hidden_states[0] = embeddings
            # hidden_states[i] = output of layer i-1
            # hidden_states[28] = output of layer 27 (final)
            
            # Use hidden state after layer exit_layer
            hidden = out.hidden_states[exit_layer + 1]  # +1 because [0] is embeddings
            
            # Apply final norm and lm_head
            hidden = self.final_norm(hidden)
            logits = self.lm_head(hidden)[0, -1]
            
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max().item()
            pred = torch.argmax(logits).item()
        
        return pred, confidence
    
    def forward_adaptive(self, token_ids: List[int],
                         thresholds: List[Tuple[int, float]] = None) -> Tuple[int, dict]:
        """
        Adaptive early exit based on confidence.
        
        thresholds: [(layer, min_confidence), ...]
        """
        if thresholds is None:
            thresholds = [(7, 0.95), (14, 0.8), (21, 0.5)]
        
        ids = torch.tensor([token_ids]).to(self.device)
        
        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True)
            
            stats = {'exit_layer': 28, 'confidences': []}
            
            for layer, min_conf in thresholds:
                hidden = out.hidden_states[layer + 1]
                hidden = self.final_norm(hidden)
                logits = self.lm_head(hidden)[0, -1]
                probs = torch.softmax(logits, dim=-1)
                confidence = probs.max().item()
                
                stats['confidences'].append((layer, confidence))
                
                if confidence >= min_conf:
                    stats['exit_layer'] = layer
                    return torch.argmax(logits).item(), stats
            
            # Use full output
            logits = out.logits[0, -1]
            return torch.argmax(logits).item(), stats


def main():
    print("=" * 70)
    print("EARLY EXIT TEST")
    print("=" * 70)
    
    model = EarlyExitTransformer()
    
    test_cases = [
        "The capital of France is",
        "Hello",
        "The quick brown",
        "Machine learning is",
        "In the year 2025",
        "The color of the sky is",
        "Water freezes at",
    ]
    
    print("\n" + "=" * 70)
    print("TEST 1: EARLY EXIT ACCURACY BY LAYER")
    print("=" * 70)
    
    for exit_layer in [3, 7, 14, 21, 27]:
        matches = 0
        for text in test_cases:
            ids = model.tokenizer(text, return_tensors="pt").input_ids[0].tolist()
            
            full_pred, _ = model.forward_full(ids)
            early_pred, _ = model.forward_early_exit(ids, exit_layer)
            
            if full_pred == early_pred:
                matches += 1
        
        print(f"  Exit at layer {exit_layer:>2}: {matches}/{len(test_cases)} match ({matches/len(test_cases)*100:.0f}%)")
    
    print("\n" + "=" * 70)
    print("TEST 2: ADAPTIVE EARLY EXIT")
    print("=" * 70)
    
    matches = 0
    layer_exits = {7: 0, 14: 0, 21: 0, 28: 0}
    
    for text in test_cases:
        ids = model.tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        
        full_pred, full_conf = model.forward_full(ids)
        adaptive_pred, stats = model.forward_adaptive(ids)
        
        exit_layer = stats['exit_layer']
        layer_exits[exit_layer] = layer_exits.get(exit_layer, 0) + 1
        
        match = "✓" if full_pred == adaptive_pred else "✗"
        if full_pred == adaptive_pred:
            matches += 1
        
        print(f"\n  '{text}'")
        print(f"    Full: {full_pred} ('{model.tokenizer.decode([full_pred])}') conf={full_conf:.3f}")
        print(f"    Adaptive: {adaptive_pred} exit@{exit_layer}")
        print(f"    Confidences: {[(l, f'{c:.3f}') for l, c in stats['confidences']]}")
        print(f"    Match: {match}")
    
    print(f"\nAdaptive accuracy: {matches}/{len(test_cases)}")
    print(f"Exit distribution: {layer_exits}")
    
    # Calculate theoretical speedup
    total_layers = sum(layer_exits.values())
    weighted_layers = sum(l * c for l, c in layer_exits.items())
    avg_layers = weighted_layers / total_layers
    speedup = 28 / avg_layers
    
    print(f"Average layers used: {avg_layers:.1f}")
    print(f"Theoretical speedup: {speedup:.2f}x")
    
    print("\n" + "=" * 70)
    print("TEST 3: TIMING (SIMULATED)")
    print("=" * 70)
    
    # Note: This doesn't give real speedup because we still run full forward
    # Real speedup requires actually stopping computation early
    
    test_text = "The capital of France is"
    ids = model.tokenizer(test_text, return_tensors="pt").input_ids[0].tolist()
    
    # Warm up
    model.forward_full(ids)
    
    # Time full
    start = time.time()
    for _ in range(10):
        model.forward_full(ids)
    full_time = (time.time() - start) / 10
    
    # Time with hidden states (overhead of capturing)
    start = time.time()
    for _ in range(10):
        model.forward_early_exit(ids, 14)
    early_time = (time.time() - start) / 10
    
    print(f"\n  Full forward:       {full_time*1000:.1f}ms")
    print(f"  With hidden states: {early_time*1000:.1f}ms")
    print(f"  Overhead: {(early_time/full_time - 1)*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
  Early exit CAN work for some tokens:
  - Layer 14 often matches layer 28 for easy tokens
  - Adaptive exit based on confidence can reduce average layers
  
  BUT:
  - Current implementation still runs full forward (just uses earlier hidden state)
  - Real speedup requires stopping computation early
  - Need custom forward pass or model modification
  
  THEORETICAL SPEEDUP:
  - If 60% exit at layer 7, 30% at layer 14, 10% at layer 28:
    Speedup = 28 / (0.6*7 + 0.3*14 + 0.1*28) = 28 / 11 = 2.5x
""")


if __name__ == "__main__":
    main()
