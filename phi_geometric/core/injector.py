"""
Knowledge Injector: Add facts to the geometric context without training.

From Doc 210: Knowledge Injection via Context Window
    Context is a "lens" that determines validity.
    Hidden states are the "focus" of that lens.
    If we inject information correctly, the model treats it as true.

Key Properties:
    - No training required
    - Facts modify the context embedding
    - Multiple injection methods (simple, authoritative, roleplay)
    - Can be used for personalization, domain adaptation, teaching

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
from typing import List, Optional
from dataclasses import dataclass

from .encoder import PHI


@dataclass
class KnowledgeFact:
    """
    A fact to inject into the geometric context.
    
    Attributes:
        content: The fact as text
        embedding: φ-encoded embedding of the fact
        weight: How strongly to inject (1.0 = normal)
        method: Injection method (simple, authoritative, roleplay)
    """
    content: str
    embedding: Optional[torch.Tensor] = None
    weight: float = 1.0
    method: str = "simple"


class KnowledgeInjector:
    """
    Inject knowledge into the geometric context.
    
    The injector modifies context embeddings to include new facts.
    This is "temporary shape modification" - the facts affect
    inference without changing the underlying weights.
    
    Example:
        injector = KnowledgeInjector()
        
        # Add facts
        injector.add_fact("Sky is typically blue")
        injector.add_fact("Grass is typically green")
        
        # Inject into context
        modified_context = injector.inject(base_context)
    
    Injection Methods:
        - simple: Direct fact embedding
        - authoritative: Weighted more heavily (like news source)
        - roleplay: Framed as character knowledge
    """
    
    def __init__(self, embedding_dim: int = 256):
        """
        Initialize the injector.
        
        Args:
            embedding_dim: Dimension of fact embeddings
        """
        self.embedding_dim = embedding_dim
        self.facts: List[KnowledgeFact] = []
    
    def add_fact(
        self, 
        content: str, 
        weight: float = 1.0,
        method: str = "simple"
    ):
        """
        Add a fact to be injected.
        
        Args:
            content: The fact as text
            weight: Injection strength (1.0 = normal)
            method: Injection method
        """
        embedding = self._text_to_embedding(content)
        
        # Apply method-specific weighting
        if method == "authoritative":
            weight *= 1.5  # Authoritative sources get more weight
        elif method == "roleplay":
            weight *= 1.2  # Roleplay is moderately weighted
        
        self.facts.append(KnowledgeFact(
            content=content,
            embedding=embedding,
            weight=weight,
            method=method
        ))
    
    def _text_to_embedding(self, text: str) -> torch.Tensor:
        """
        Convert text to a φ-encoded embedding.
        
        This is a simple hash-based embedding. In practice,
        you would use a proper text encoder.
        
        Args:
            text: Input text
            
        Returns:
            φ-encoded embedding tensor
        """
        embedding = torch.zeros(self.embedding_dim)
        
        # Simple character-based embedding
        for i, char in enumerate(text):
            idx = (ord(char) * (i + 1)) % self.embedding_dim
            level = (ord(char) % 20) - 10
            embedding[idx] += PHI ** level
        
        # Normalize
        norm = embedding.norm()
        if norm > 1e-10:
            embedding = embedding / norm
        
        return embedding
    
    def inject(self, base_context: torch.Tensor) -> torch.Tensor:
        """
        Inject all facts into the base context.
        
        The injection blends fact embeddings with the base context,
        modifying it to include the injected knowledge.
        
        Args:
            base_context: Original context embedding
            
        Returns:
            Modified context with injected facts
        """
        if not self.facts:
            return base_context
        
        # Ensure base_context is at least 1D
        if base_context.dim() == 0:
            base_context = base_context.unsqueeze(0)
        
        # Combine fact embeddings
        fact_embedding = torch.zeros_like(base_context)
        total_weight = 0
        
        for fact in self.facts:
            if fact.embedding is not None:
                # Resize embedding to match context
                emb = self._resize_embedding(fact.embedding, base_context.shape)
                fact_embedding = fact_embedding + emb * fact.weight
                total_weight += fact.weight
        
        if total_weight > 0:
            fact_embedding = fact_embedding / total_weight
        
        # Blend: base context + injected facts
        # The blend ratio determines how much facts override
        blend_ratio = min(0.5, total_weight / (total_weight + 1))
        injected = (1 - blend_ratio) * base_context + blend_ratio * fact_embedding
        
        return injected
    
    def _resize_embedding(
        self, 
        embedding: torch.Tensor, 
        target_shape: torch.Size
    ) -> torch.Tensor:
        """Resize embedding to match target shape."""
        result = torch.zeros(target_shape)
        
        # Handle different dimensionalities
        if embedding.dim() == 1:
            min_dim = min(embedding.shape[0], target_shape[-1])
            if result.dim() == 1:
                result[:min_dim] = embedding[:min_dim]
            else:
                result[..., :min_dim] = embedding[:min_dim]
        else:
            # Multi-dimensional: take what fits
            slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(embedding.shape, target_shape))
            result[slices] = embedding[slices]
        
        return result
    
    def clear(self):
        """Clear all injected facts."""
        self.facts = []
    
    def get_facts(self) -> List[str]:
        """Get list of fact contents."""
        return [f.content for f in self.facts]
    
    def num_facts(self) -> int:
        """Get number of injected facts."""
        return len(self.facts)


def test_injector():
    """Test the knowledge injector."""
    print("=" * 60)
    print("KNOWLEDGE INJECTOR TEST")
    print("=" * 60)
    
    injector = KnowledgeInjector(embedding_dim=64)
    
    # Test 1: Add facts
    print("\n1. Adding facts:")
    injector.add_fact("Sky is typically blue")
    injector.add_fact("Grass is typically green")
    injector.add_fact("Fire is hot", method="authoritative")
    print(f"   Added {injector.num_facts()} facts")
    
    # Test 2: Inject into context
    print("\n2. Injecting into context:")
    base_context = torch.randn(64)
    modified = injector.inject(base_context)
    
    # Measure change
    diff = (modified - base_context).norm() / base_context.norm()
    print(f"   Context change: {diff:.3f}")
    
    # Test 3: Different methods
    print("\n3. Injection methods:")
    for method in ["simple", "authoritative", "roleplay"]:
        inj = KnowledgeInjector(embedding_dim=64)
        inj.add_fact("Test fact", method=method)
        mod = inj.inject(base_context)
        diff = (mod - base_context).norm() / base_context.norm()
        print(f"   {method}: change = {diff:.3f}")
    
    # Test 4: Multiple facts
    print("\n4. Multiple facts:")
    inj = KnowledgeInjector(embedding_dim=64)
    for i in range(10):
        inj.add_fact(f"Fact number {i}")
    mod = inj.inject(base_context)
    diff = (mod - base_context).norm() / base_context.norm()
    print(f"   10 facts: change = {diff:.3f}")
    
    print("\n" + "=" * 60)
    print("KNOWLEDGE INJECTOR TEST COMPLETE")
    print("=" * 60)
    
    return injector


if __name__ == "__main__":
    test_injector()
