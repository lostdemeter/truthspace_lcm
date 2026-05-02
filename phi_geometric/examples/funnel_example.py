"""
Funnel Pattern Example: Classification

The Funnel pattern is convergent (many → one).
Used for: Classification, regression, depth estimation.

Observed in: DA2 (Depth Anything V2)

Characteristics:
    - Simple, focused, one output per location
    - No self-reference
    - Extremely compressible

This example builds a classifier without training.

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
from typing import List, Optional

from ..core import (
    GeometricAI, ProblemSpec, IOSpec, DataType,
    PhiEncoder, Funnel
)


class FunnelClassifier:
    """
    A classifier using the Funnel pattern.
    
    The Funnel pattern converges many input features to
    a single class prediction. This is the simplest pattern.
    
    Example:
        classifier = FunnelClassifier(
            input_dim=64,
            num_classes=10,
            class_names=["cat", "dog", "bird", ...]
        )
        
        # Inject knowledge about classes
        classifier.inject_class_knowledge("cat", "small furry animal")
        classifier.inject_class_knowledge("dog", "loyal companion")
        
        # Classify
        class_idx, confidence = classifier.classify(features)
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        num_classes: int = 10,
        hidden_dim: int = 32,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize the classifier.
        
        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension
            class_names: Optional names for classes
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        
        # Create problem specification
        self.problem = ProblemSpec(
            name="funnel_classifier",
            inputs=[IOSpec("features", DataType.VECTOR, (input_dim,), "input features")],
            outputs=[IOSpec("logits", DataType.VECTOR, (num_classes,), "class logits")],
        )
        
        # Create GeometricAI
        self.ai = GeometricAI(self.problem)
        
        # Inject default knowledge
        self._inject_default_knowledge()
    
    def _inject_default_knowledge(self):
        """Inject default knowledge about classification."""
        self.ai.inject_knowledge("Classes are mutually exclusive")
        self.ai.inject_knowledge("Higher logits indicate higher confidence")
    
    def inject_class_knowledge(self, class_name: str, description: str):
        """
        Inject knowledge about a specific class.
        
        Args:
            class_name: Name of the class
            description: Description of what the class represents
        """
        self.ai.inject_knowledge(f"{class_name}: {description}")
    
    def classify(self, features: torch.Tensor) -> tuple:
        """
        Classify input features.
        
        Args:
            features: Input features [input_dim] or [B, input_dim]
            
        Returns:
            (class_index, confidence)
        """
        logits = self.ai(features)
        
        # Softmax for probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get top class
        confidence, class_idx = probs.max(dim=-1)
        
        return class_idx.item(), confidence.item()
    
    def classify_with_name(self, features: torch.Tensor) -> tuple:
        """
        Classify and return class name.
        
        Args:
            features: Input features
            
        Returns:
            (class_name, confidence)
        """
        class_idx, confidence = self.classify(features)
        class_name = self.class_names[class_idx]
        return class_name, confidence
    
    def top_k(self, features: torch.Tensor, k: int = 5) -> List[tuple]:
        """
        Get top-k predictions.
        
        Args:
            features: Input features
            k: Number of top predictions
            
        Returns:
            List of (class_name, probability)
        """
        logits = self.ai(features)
        probs = torch.softmax(logits, dim=-1)
        
        top_probs, top_indices = probs.topk(k, dim=-1)
        
        results = []
        for i in range(k):
            idx = top_indices[i].item() if top_indices.dim() > 0 else top_indices.item()
            prob = top_probs[i].item() if top_probs.dim() > 0 else top_probs.item()
            results.append((self.class_names[idx], prob))
        
        return results
    
    def stats(self):
        """Get classifier statistics."""
        return self.ai.stats()


def demo_funnel_classifier():
    """Demonstrate the Funnel classifier."""
    print("=" * 70)
    print("FUNNEL PATTERN EXAMPLE: Classification")
    print("=" * 70)
    
    # Create classifier
    classifier = FunnelClassifier(
        input_dim=64,
        num_classes=5,
        class_names=["cat", "dog", "bird", "fish", "rabbit"]
    )
    
    # Inject knowledge about each class
    classifier.inject_class_knowledge("cat", "small furry feline")
    classifier.inject_class_knowledge("dog", "loyal canine companion")
    classifier.inject_class_knowledge("bird", "feathered flying creature")
    classifier.inject_class_knowledge("fish", "aquatic gill-breathing animal")
    classifier.inject_class_knowledge("rabbit", "hopping long-eared mammal")
    
    print("\nClassifier created:")
    print(f"  Input dim: {classifier.input_dim}")
    print(f"  Classes: {classifier.class_names}")
    print(f"  Pattern: Funnel (convergent)")
    
    # Test classification
    print("\n--- Classification Tests ---")
    
    for i in range(5):
        features = torch.randn(64)
        class_name, confidence = classifier.classify_with_name(features)
        print(f"  Sample {i+1}: {class_name} (confidence: {confidence:.3f})")
    
    # Top-k predictions
    print("\n--- Top-3 Predictions ---")
    features = torch.randn(64)
    top3 = classifier.top_k(features, k=3)
    for name, prob in top3:
        print(f"  {name}: {prob:.3f}")
    
    # Stats
    print("\n--- Statistics ---")
    stats = classifier.stats()
    print(f"  Memory size: {stats['memory_size']}")
    print(f"  Hit rate: {stats['memory_hit_rate']:.1%}")
    print(f"  Inferences: {stats['inference_count']}")
    
    print("\n" + "=" * 70)
    print("FUNNEL EXAMPLE COMPLETE")
    print("=" * 70)
    
    return classifier


if __name__ == "__main__":
    demo_funnel_classifier()
