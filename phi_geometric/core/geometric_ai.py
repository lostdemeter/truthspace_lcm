"""
GeometricAI: Unified interface for geometric neural computation.

This is the main entry point for the φ-Geometric Framework.
It combines all four components:
    1. Shape Projection (from problem structure)
    2. Knowledge Injection (context as lens)
    3. Signature Memory (self-assembling)
    4. Bottleneck Filter (φ-validity)

No statistical training required!

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
from typing import Dict, Optional, Any, List

from .encoder import PhiEncoder
from .patterns import Pattern
from .projector import ShapeProjector, ProblemSpec
from .navigator import Navigator
from .memory import SignatureMemory
from .injector import KnowledgeInjector
from .filter import BottleneckFilter


class GeometricAI:
    """
    Unified Geometric AI: Build and run AI without training.
    
    GeometricAI combines four components:
        1. Shape Projection: Derive φ-coordinates from problem structure
        2. Knowledge Injection: Add facts via context modification
        3. Signature Memory: Self-assembling cache for fast lookup
        4. Bottleneck Filter: Validate outputs through φ-constraint
    
    Example:
        from phi_geometric import GeometricAI, ProblemSpec, IOSpec, DataType
        
        # Define problem
        problem = ProblemSpec(
            name="classifier",
            inputs=[IOSpec("features", DataType.VECTOR, (64,))],
            outputs=[IOSpec("class", DataType.VECTOR, (10,))],
        )
        
        # Create AI (no training!)
        ai = GeometricAI(problem)
        
        # Inject knowledge
        ai.inject_knowledge("Class 0 is for small values")
        ai.inject_knowledge("Class 9 is for large values")
        
        # Run inference
        output = ai(input_tensor)
        
        # Check stats
        print(ai.stats())
    
    Attributes:
        problem: The problem specification
        pattern: The navigation pattern (Funnel, Spiral, Web, etc.)
        encoder: PhiEncoder for φ-basis operations
        navigator: Navigator for geometric traversal
        memory: SignatureMemory for caching
        injector: KnowledgeInjector for fact injection
        filter: BottleneckFilter for validity checking
    """
    
    def __init__(
        self,
        problem: ProblemSpec,
        K: int = 32,
        memory_threshold: float = 0.5,
        validity_tolerance: float = 0.3,
        device: str = "cpu"
    ):
        """
        Initialize GeometricAI.
        
        Args:
            problem: Problem specification
            K: φ-encoding resolution
            memory_threshold: Signature distance for cache hit
            validity_tolerance: φ-distance for validity
            device: Compute device
        """
        self.problem = problem
        self.device = device
        
        # Component 1: Encoder
        self.encoder = PhiEncoder(K=K)
        
        # Component 2: Project shape
        self.projector = ShapeProjector(self.encoder)
        self.pattern, self.phi_weights = self.projector.project(problem)
        
        # Component 3: Navigator
        self.navigator = Navigator(
            self.pattern, 
            self.phi_weights, 
            self.encoder,
            device=device
        )
        
        # Component 4: Knowledge injection
        self.injector = KnowledgeInjector(
            embedding_dim=256
        )
        
        # Component 5: Signature memory
        self.memory = SignatureMemory(
            threshold=memory_threshold
        )
        
        # Component 6: Bottleneck filter
        self.filter = BottleneckFilter(
            tolerance=validity_tolerance
        )
        
        # Statistics
        self._inference_count = 0
        
        print(f"GeometricAI initialized:")
        print(f"  Problem: {problem.name}")
        print(f"  Pattern: {self.pattern.name} ({self.pattern.topology.value})")
        print(f"  Nodes: {len(self.pattern.nodes)}")
        print(f"  Weights: {len(self.phi_weights)} tensors")
    
    def inject_knowledge(
        self, 
        fact: str, 
        weight: float = 1.0,
        method: str = "simple"
    ):
        """
        Inject a fact into the knowledge base.
        
        Args:
            fact: The fact to inject
            weight: Injection strength
            method: Injection method (simple, authoritative, roleplay)
        """
        self.injector.add_fact(fact, weight, method)
    
    def forward(
        self, 
        input_tensor: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        use_memory: bool = True,
        validate: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through the geometric AI.
        
        Steps:
            1. Check memory for cached result
            2. If miss, inject knowledge into context
            3. Navigate through the shape
            4. Filter through bottleneck
            5. Store in memory
        
        Args:
            input_tensor: Input data
            context: Optional context for cross-attention
            use_memory: Whether to use signature memory
            validate: Whether to validate through bottleneck
            
        Returns:
            Output tensor
        """
        self._inference_count += 1
        
        # Step 1: Check memory
        if use_memory:
            cached, distance = self.memory.lookup(input_tensor)
            if cached is not None:
                return cached
        
        # Step 2: Prepare input with knowledge injection
        x = input_tensor.float().to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Inject knowledge into context
        x_injected = self.injector.inject(x)
        
        # Step 3: Navigate through the shape
        output = self.navigator.navigate(x_injected, context)
        
        # Step 4: Validate through bottleneck
        if validate:
            is_valid, phi_level = self.filter.is_valid(output)
            if not is_valid:
                output = self.filter.adjust_for_validity(output)
        
        # Step 5: Store in memory
        if use_memory:
            self.memory.store(input_tensor, output)
        
        # Squeeze if input was 1D
        if input_tensor.dim() == 1 and output.dim() == 2 and output.shape[0] == 1:
            output = output.squeeze(0)
        
        return output
    
    def __call__(
        self, 
        input_tensor: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Alias for forward."""
        return self.forward(input_tensor, **kwargs)
    
    def batch_forward(
        self, 
        inputs: List[torch.Tensor],
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Process a batch of inputs.
        
        Args:
            inputs: List of input tensors
            **kwargs: Passed to forward()
            
        Returns:
            List of output tensors
        """
        return [self.forward(x, **kwargs) for x in inputs]
    
    def validate_output(self, output: torch.Tensor) -> Dict[str, Any]:
        """
        Validate an output through the bottleneck.
        
        Args:
            output: Output tensor to validate
            
        Returns:
            Dict with is_valid, phi_level, score
        """
        is_valid, phi_level = self.filter.is_valid(output)
        score = self.filter.validity_score(output)
        
        return {
            'is_valid': is_valid,
            'phi_level': phi_level,
            'score': score,
            'target_phi': self.filter.target_phi,
            'tolerance': self.filter.tolerance,
        }
    
    def stats(self) -> Dict[str, Any]:
        """
        Get statistics about the geometric AI.
        
        Returns:
            Dict with problem, pattern, memory, and inference stats
        """
        return {
            'problem': self.problem.name,
            'pattern': self.pattern.name,
            'topology': self.pattern.topology.value,
            'num_nodes': len(self.pattern.nodes),
            'num_weights': len(self.phi_weights),
            'num_facts': self.injector.num_facts(),
            'memory_size': self.memory.size(),
            'memory_hit_rate': self.memory.hit_rate(),
            'inference_count': self._inference_count,
        }
    
    def describe(self) -> str:
        """Human-readable description."""
        s = self.stats()
        return (
            f"GeometricAI: {s['problem']}\n"
            f"  Pattern: {s['pattern']} ({s['topology']})\n"
            f"  Nodes: {s['num_nodes']}, Weights: {s['num_weights']}\n"
            f"  Facts: {s['num_facts']}, Memory: {s['memory_size']}\n"
            f"  Hit rate: {s['memory_hit_rate']:.1%}\n"
            f"  Inferences: {s['inference_count']}"
        )
    
    def clear_memory(self):
        """Clear the signature memory."""
        self.memory.clear()
    
    def clear_knowledge(self):
        """Clear injected knowledge."""
        self.injector.clear()
    
    def reset(self):
        """Reset memory and knowledge."""
        self.clear_memory()
        self.clear_knowledge()
        self._inference_count = 0


def test_geometric_ai():
    """Test the GeometricAI class."""
    print("=" * 70)
    print("GEOMETRIC AI TEST")
    print("=" * 70)
    
    from .projector import IOSpec, DataType
    
    # Test 1: Simple classifier
    print("\n1. Simple Classifier:")
    problem = ProblemSpec(
        name="classifier",
        inputs=[IOSpec("features", DataType.VECTOR, (64,))],
        outputs=[IOSpec("class", DataType.VECTOR, (10,))],
    )
    
    ai = GeometricAI(problem)
    ai.inject_knowledge("Class 0 is for small values")
    ai.inject_knowledge("Class 9 is for large values")
    
    # Run inference
    x = torch.randn(64)
    y = ai(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {y.shape}")
    print(f"   Predicted class: {y.argmax().item()}")
    
    # Test memory
    y2 = ai(x)  # Should hit cache
    print(f"   Memory hit rate: {ai.memory.hit_rate():.1%}")
    
    # Test 2: Colorizer
    print("\n2. Colorizer:")
    problem = ProblemSpec(
        name="colorizer",
        inputs=[IOSpec("gray", DataType.IMAGE, (8, 8, 1))],
        outputs=[IOSpec("color", DataType.IMAGE, (8, 8, 2))],
    )
    
    ai = GeometricAI(problem)
    ai.inject_knowledge("Sky is blue")
    ai.inject_knowledge("Grass is green")
    
    x = torch.randn(64)  # Flattened 8x8x1
    y = ai(x)
    validation = ai.validate_output(y)
    print(f"   Output shape: {y.shape}")
    print(f"   φ-level: {validation['phi_level']:.3f}")
    print(f"   Valid: {validation['is_valid']}")
    
    # Test 3: Language model
    print("\n3. Language Model:")
    problem = ProblemSpec(
        name="language",
        inputs=[IOSpec("tokens", DataType.SEQUENCE, (32,))],
        outputs=[IOSpec("next", DataType.VECTOR, (100,))],
        temporal=True,
    )
    
    ai = GeometricAI(problem)
    ai.inject_knowledge("Common words are more likely")
    
    x = torch.randn(32)
    y = ai(x)
    print(f"   Output shape: {y.shape}")
    print(f"   Top token: {y.argmax().item()}")
    
    # Test 4: Stats
    print("\n4. Statistics:")
    print(ai.describe())
    
    print("\n" + "=" * 70)
    print("GEOMETRIC AI TEST COMPLETE")
    print("=" * 70)
    
    return ai


if __name__ == "__main__":
    test_geometric_ai()
