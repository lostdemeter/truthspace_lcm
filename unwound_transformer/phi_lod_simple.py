#!/usr/bin/env python3
"""
Simple LOD Test: Skip Layers for Easy Tokens
=============================================

Instead of complex per-weight LOD, test a simpler approach:
- Easy tokens: skip some layers entirely
- Hard tokens: run all layers

This is based on the finding that Layer 0 is 96% linear.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import List, Tuple
import time

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)


def phi_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    phi_powers = PHI ** ((x - x_max) / LN_PHI)
    return phi_powers / np.sum(phi_powers, axis=axis, keepdims=True)


class SimpleLODTransformer:
    """
    Simple LOD: Control which layers to run.
    
    Based on findings:
    - Layer 0 is 96% in linear regime
    - Layers 0-2 are "pre-click" (Doc 189)
    - Layer 3 is the "click" point
    - Layers 4+ are "post-click"
    """
    
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
    
    def forward_skip_layers(self, token_ids: List[int], 
                            skip_layers: List[int]) -> Tuple[int, float]:
        """
        Forward pass skipping specified layers.
        
        This tests if we can skip early layers for easy tokens.
        """
        ids = torch.tensor([token_ids]).to(self.device)
        seq_len = ids.shape[1]
        
        with torch.no_grad():
            # Get embeddings
            hidden = self.model.model.embed_tokens(ids)
            
            # Create position embeddings using model's rotary_emb
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
            rotary_emb = self.model.model.rotary_emb
            cos, sin = rotary_emb(hidden, position_ids)
            position_embeddings = (cos, sin)
            
            # Run through layers, skipping some
            for i, layer in enumerate(self.model.model.layers):
                if i in skip_layers:
                    continue  # Skip this layer
                hidden = layer(hidden, position_embeddings=position_embeddings)[0]
            
            # Final norm
            hidden = self.model.model.norm(hidden)
            
            # LM head
            logits = self.model.lm_head(hidden)[0, -1]
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max().item()
            pred = torch.argmax(logits).item()
        
        return pred, confidence
    
    def forward_early_exit(self, token_ids: List[int], 
                           exit_layer: int) -> Tuple[int, float]:
        """
        Early exit: stop at specified layer.
        
        This tests if we can exit early for easy tokens.
        """
        ids = torch.tensor([token_ids]).to(self.device)
        seq_len = ids.shape[1]
        
        with torch.no_grad():
            hidden = self.model.model.embed_tokens(ids)
            
            # Create position embeddings
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
            rotary_emb = self.model.model.rotary_emb
            cos, sin = rotary_emb(hidden, position_ids)
            position_embeddings = (cos, sin)
            
            for i, layer in enumerate(self.model.model.layers):
                if i >= exit_layer:
                    break
                hidden = layer(hidden, position_embeddings=position_embeddings)[0]
            
            hidden = self.model.model.norm(hidden)
            logits = self.model.lm_head(hidden)[0, -1]
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max().item()
            pred = torch.argmax(logits).item()
        
        return pred, confidence


def main():
    print("=" * 70)
    print("SIMPLE LOD TEST: LAYER SKIPPING")
    print("=" * 70)
    
    model = SimpleLODTransformer()
    
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
    print("TEST 1: FULL vs SKIP EARLY LAYERS")
    print("=" * 70)
    
    # Test skipping layers 0-2 (pre-click)
    skip_layers = [0, 1, 2]
    
    matches = 0
    for text in test_cases:
        ids = model.tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        
        full_pred, full_conf = model.forward_full(ids)
        skip_pred, skip_conf = model.forward_skip_layers(ids, skip_layers)
        
        match = "✓" if full_pred == skip_pred else "✗"
        if full_pred == skip_pred:
            matches += 1
        
        print(f"\n  '{text}'")
        print(f"    Full:  {full_pred} ('{model.tokenizer.decode([full_pred])}') conf={full_conf:.3f}")
        print(f"    Skip:  {skip_pred} ('{model.tokenizer.decode([skip_pred])}') conf={skip_conf:.3f}")
        print(f"    Match: {match}")
    
    print(f"\nSkip layers {skip_layers}: {matches}/{len(test_cases)} match")
    
    print("\n" + "=" * 70)
    print("TEST 2: EARLY EXIT")
    print("=" * 70)
    
    # Test early exit at different layers
    for exit_layer in [7, 14, 21, 28]:
        matches = 0
        for text in test_cases:
            ids = model.tokenizer(text, return_tensors="pt").input_ids[0].tolist()
            
            full_pred, _ = model.forward_full(ids)
            early_pred, early_conf = model.forward_early_exit(ids, exit_layer)
            
            if full_pred == early_pred:
                matches += 1
        
        print(f"  Exit at layer {exit_layer:>2}: {matches}/{len(test_cases)} match")
    
    print("\n" + "=" * 70)
    print("TEST 3: TIMING")
    print("=" * 70)
    
    test_text = "The capital of France is"
    ids = model.tokenizer(test_text, return_tensors="pt").input_ids[0].tolist()
    
    # Warm up
    model.forward_full(ids)
    
    # Time full
    start = time.time()
    for _ in range(10):
        model.forward_full(ids)
    full_time = (time.time() - start) / 10
    
    # Time skip
    start = time.time()
    for _ in range(10):
        model.forward_skip_layers(ids, [0, 1, 2])
    skip_time = (time.time() - start) / 10
    
    # Time early exit at layer 14
    start = time.time()
    for _ in range(10):
        model.forward_early_exit(ids, 14)
    early_time = (time.time() - start) / 10
    
    print(f"\n  Full (28 layers):    {full_time*1000:.1f}ms")
    print(f"  Skip 0-2 (25 layers): {skip_time*1000:.1f}ms ({full_time/skip_time:.2f}x)")
    print(f"  Early exit (14):      {early_time*1000:.1f}ms ({full_time/early_time:.2f}x)")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
  Layer skipping/early exit can provide speedup, but accuracy depends
  on which layers are skipped and the token difficulty.
  
  Key findings:
  - Skipping layers 0-2 may preserve accuracy for some tokens
  - Early exit at layer 14 gives ~2x speedup but may lose accuracy
  - Need adaptive approach: easy tokens exit early, hard tokens run full
""")


if __name__ == "__main__":
    main()
