"""
Spiral Pattern Example: Language Modeling

The Spiral pattern is self-referential (deep with attention).
Used for: Language modeling, reasoning, sequential tasks.

Observed in: Qwen2-7B

Characteristics:
    - Many identical segments (layers)
    - Self-attention at every layer
    - MESH principle applies

This example builds a simple language model without training.

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
from typing import List, Optional, Dict

from ..core import (
    GeometricAI, ProblemSpec, IOSpec, DataType,
    PhiEncoder, Spiral
)


class SpiralLanguageModel:
    """
    A language model using the Spiral pattern.
    
    The Spiral pattern is self-referential - each layer
    attends to itself, building up context through depth.
    
    Example:
        lm = SpiralLanguageModel(
            vocab_size=1000,
            context_length=64,
            layers=4
        )
        
        # Inject language knowledge
        lm.inject_knowledge("Common words are more likely")
        lm.inject_knowledge("Sentences end with punctuation")
        
        # Predict next token
        next_token = lm.predict_next(token_embeddings)
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        context_length: int = 64,
        dim: int = 256,
        layers: int = 4,
        heads: int = 4
    ):
        """
        Initialize the language model.
        
        Args:
            vocab_size: Size of vocabulary
            context_length: Maximum context length
            dim: Hidden dimension
            layers: Number of transformer layers
            heads: Number of attention heads
        """
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.dim = dim
        self.layers = layers
        self.heads = heads
        
        # Create problem specification - use dim for input to match internal dimensions
        self.problem = ProblemSpec(
            name="spiral_language_model",
            inputs=[IOSpec("tokens", DataType.VECTOR, (dim,), "token embeddings")],
            outputs=[IOSpec("logits", DataType.VECTOR, (vocab_size,), "next token logits")],
            temporal=True,
            causal=True
        )
        
        # Create GeometricAI with appropriate dimensions
        self.ai = GeometricAI(self.problem, memory_threshold=1.0)
        
        # Simple vocabulary (for demo)
        self.vocab = {i: f"token_{i}" for i in range(vocab_size)}
        self.vocab[0] = "<pad>"
        self.vocab[1] = "<unk>"
        self.vocab[2] = "<eos>"
        
        # Inject default knowledge
        self._inject_default_knowledge()
    
    def _inject_default_knowledge(self):
        """Inject default language knowledge."""
        self.ai.inject_knowledge("Language follows grammatical patterns")
        self.ai.inject_knowledge("Context determines meaning")
        self.ai.inject_knowledge("Common words appear more frequently")
    
    def inject_knowledge(self, fact: str):
        """Inject language knowledge."""
        self.ai.inject_knowledge(fact)
    
    def predict_next(self, embeddings: torch.Tensor) -> int:
        """
        Predict the next token.
        
        Args:
            embeddings: Token embeddings [context_length, dim] or [context_length]
            
        Returns:
            Next token index
        """
        logits = self.ai(embeddings)
        
        # Get top prediction
        next_token = logits.argmax(dim=-1)
        
        return next_token.item() if next_token.dim() == 0 else next_token[-1].item()
    
    def predict_top_k(self, embeddings: torch.Tensor, k: int = 5) -> List[tuple]:
        """
        Get top-k next token predictions.
        
        Args:
            embeddings: Token embeddings
            k: Number of predictions
            
        Returns:
            List of (token, probability)
        """
        logits = self.ai(embeddings)
        probs = torch.softmax(logits, dim=-1)
        
        # Handle multi-dimensional output
        if probs.dim() > 1:
            probs = probs[-1]  # Take last position
        
        top_probs, top_indices = probs.topk(min(k, len(probs)))
        
        results = []
        for i in range(len(top_indices)):
            idx = top_indices[i].item()
            prob = top_probs[i].item()
            token = self.vocab.get(idx, f"token_{idx}")
            results.append((token, prob))
        
        return results
    
    def generate(
        self, 
        prompt_embeddings: torch.Tensor, 
        max_tokens: int = 10
    ) -> List[int]:
        """
        Generate a sequence of tokens.
        
        Args:
            prompt_embeddings: Initial embeddings
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of generated token indices
        """
        generated = []
        current = prompt_embeddings.clone()
        
        for _ in range(max_tokens):
            next_token = self.predict_next(current)
            generated.append(next_token)
            
            # Stop at EOS
            if next_token == 2:  # <eos>
                break
            
            # Update context (simplified - just use same embeddings)
            # In practice, would embed the new token
        
        return generated
    
    def stats(self):
        """Get model statistics."""
        return self.ai.stats()


def demo_spiral_language_model():
    """Demonstrate the Spiral language model."""
    print("=" * 70)
    print("SPIRAL PATTERN EXAMPLE: Language Modeling")
    print("=" * 70)
    
    # Create language model
    lm = SpiralLanguageModel(
        vocab_size=100,
        context_length=32,
        dim=64,
        layers=4,
        heads=4
    )
    
    # Inject knowledge
    lm.inject_knowledge("The word 'the' is very common")
    lm.inject_knowledge("Verbs follow subjects")
    lm.inject_knowledge("Questions end with question marks")
    
    print("\nLanguage Model created:")
    print(f"  Vocab size: {lm.vocab_size}")
    print(f"  Context length: {lm.context_length}")
    print(f"  Layers: {lm.layers}")
    print(f"  Pattern: Spiral (self-referential)")
    
    # Test prediction
    print("\n--- Next Token Prediction ---")
    
    for i in range(3):
        embeddings = torch.randn(32)  # Simplified embeddings
        next_token = lm.predict_next(embeddings)
        print(f"  Prediction {i+1}: token_{next_token}")
    
    # Top-k predictions
    print("\n--- Top-5 Predictions ---")
    embeddings = torch.randn(32)
    top5 = lm.predict_top_k(embeddings, k=5)
    for token, prob in top5:
        print(f"  {token}: {prob:.4f}")
    
    # Generation
    print("\n--- Token Generation ---")
    prompt = torch.randn(32)
    generated = lm.generate(prompt, max_tokens=5)
    print(f"  Generated tokens: {generated}")
    
    # Stats
    print("\n--- Statistics ---")
    stats = lm.stats()
    print(f"  Pattern: {stats['pattern']}")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Memory hit rate: {stats['memory_hit_rate']:.1%}")
    
    print("\n" + "=" * 70)
    print("SPIRAL EXAMPLE COMPLETE")
    print("=" * 70)
    
    return lm


if __name__ == "__main__":
    demo_spiral_language_model()
