"""
Braid Pattern Example: Multi-Modal Fusion

The Braid pattern is intertwined (multiple streams that cross).
Used for: Vision-language models, audio-visual understanding, sensor fusion.

Characteristics:
    - Multiple parallel streams
    - Periodic cross-attention between streams
    - Each stream maintains its own representation

This example builds a vision-language model without training.

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
from typing import Dict, List, Optional, Tuple

from ..core import (
    GeometricAI, ProblemSpec, IOSpec, DataType,
    PhiEncoder, Braid
)


class BraidMultiModal:
    """
    A multi-modal model using the Braid pattern.
    
    The Braid pattern processes multiple modalities in parallel,
    with periodic cross-attention allowing information exchange.
    
    Example:
        model = BraidMultiModal(
            modalities=["vision", "language"],
            dim=256,
            layers=6
        )
        
        # Inject modality knowledge
        model.inject_knowledge("vision", "Visual features describe appearance")
        model.inject_knowledge("language", "Text provides semantic context")
        
        # Fuse modalities
        output = model.fuse(vision_features, language_features)
    """
    
    def __init__(
        self,
        modalities: Optional[List[str]] = None,
        dim: int = 256,
        layers: int = 6,
        cross_every: int = 2
    ):
        """
        Initialize the multi-modal model.
        
        Args:
            modalities: List of modality names
            dim: Hidden dimension per modality
            layers: Number of layers
            cross_every: Cross-attention frequency
        """
        self.modalities = modalities or ["vision", "language"]
        self.dim = dim
        self.layers = layers
        self.cross_every = cross_every
        
        # Create problem specification
        inputs = [
            IOSpec(mod, DataType.VECTOR, (dim,), f"{mod} features")
            for mod in self.modalities
        ]
        
        self.problem = ProblemSpec(
            name="braid_multimodal",
            inputs=inputs,
            outputs=[IOSpec("fused", DataType.VECTOR, (dim,), "fused representation")],
            cross_modal=True
        )
        
        # Create GeometricAI
        self.ai = GeometricAI(self.problem)
        
        # Inject default knowledge
        self._inject_default_knowledge()
    
    def _inject_default_knowledge(self):
        """Inject default multi-modal knowledge."""
        self.ai.inject_knowledge("Modalities provide complementary information")
        self.ai.inject_knowledge("Cross-modal attention aligns representations")
        for mod in self.modalities:
            self.ai.inject_knowledge(f"{mod} stream processes {mod} features")
    
    def inject_knowledge(self, modality: str, fact: str):
        """
        Inject knowledge about a modality.
        
        Args:
            modality: Which modality
            fact: Knowledge to inject
        """
        self.ai.inject_knowledge(f"{modality}: {fact}")
    
    def fuse(self, *modality_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse multiple modality features.
        
        Args:
            *modality_features: Features for each modality
            
        Returns:
            Fused representation
        """
        if len(modality_features) != len(self.modalities):
            raise ValueError(f"Expected {len(self.modalities)} modalities, got {len(modality_features)}")
        
        # Concatenate all modality features
        combined = torch.cat([f.flatten() for f in modality_features])
        
        # Run through geometric AI
        output = self.ai(combined)
        
        return output
    
    def fuse_dict(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse modality features from a dict.
        
        Args:
            features: Dict of {modality: features}
            
        Returns:
            Fused representation
        """
        ordered_features = [features[mod] for mod in self.modalities]
        return self.fuse(*ordered_features)
    
    def stats(self):
        """Get model statistics."""
        return self.ai.stats()


def demo_braid_multimodal():
    """Demonstrate the Braid multi-modal model."""
    print("=" * 70)
    print("BRAID PATTERN EXAMPLE: Multi-Modal Fusion")
    print("=" * 70)
    
    # Create multi-modal model
    model = BraidMultiModal(
        modalities=["vision", "language", "audio"],
        dim=64,
        layers=4,
        cross_every=2
    )
    
    # Inject modality knowledge
    model.inject_knowledge("vision", "Captures spatial and appearance information")
    model.inject_knowledge("language", "Provides semantic and contextual meaning")
    model.inject_knowledge("audio", "Conveys temporal and acoustic patterns")
    
    print("\nMulti-Modal Model created:")
    print(f"  Modalities: {model.modalities}")
    print(f"  Dimension: {model.dim}")
    print(f"  Layers: {model.layers}")
    print(f"  Cross every: {model.cross_every} layers")
    print(f"  Pattern: Braid (intertwined)")
    
    # Test fusion
    print("\n--- Multi-Modal Fusion ---")
    
    vision_features = torch.randn(64)
    language_features = torch.randn(64)
    audio_features = torch.randn(64)
    
    fused = model.fuse(vision_features, language_features, audio_features)
    print(f"  Vision: {vision_features.shape}")
    print(f"  Language: {language_features.shape}")
    print(f"  Audio: {audio_features.shape}")
    print(f"  Fused: {fused.shape}")
    
    # Dict-based fusion
    print("\n--- Dict-Based Fusion ---")
    features = {
        "vision": torch.randn(64),
        "language": torch.randn(64),
        "audio": torch.randn(64)
    }
    fused = model.fuse_dict(features)
    print(f"  Fused from dict: {fused.shape}")
    
    # Stats
    print("\n--- Statistics ---")
    stats = model.stats()
    print(f"  Pattern: {stats['pattern']}")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Facts: {stats['num_facts']}")
    print(f"  Memory hit rate: {stats['memory_hit_rate']:.1%}")
    
    print("\n" + "=" * 70)
    print("BRAID EXAMPLE COMPLETE")
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    demo_braid_multimodal()
