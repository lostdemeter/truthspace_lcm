"""
Navigator: Traverse shapes on the φ-lattice for inference.

The navigator executes geometric navigation through a pattern,
using φ-encoded weights to compute outputs.

Key Principles:
    - Navigation IS computation
    - The shape IS the knowledge
    - Traversal follows pattern topology

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .encoder import PhiEncoder
from .patterns import Pattern, Topology


class Navigator:
    """
    Navigate through a φ-shape for inference.
    
    The navigator traverses the pattern topology, applying
    φ-encoded weights at each node to transform the input.
    
    Example:
        navigator = Navigator(pattern, phi_weights, encoder)
        output = navigator.navigate(input_tensor)
    """
    
    def __init__(
        self,
        pattern: Pattern,
        phi_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        encoder: PhiEncoder,
        device: str = "cpu"
    ):
        """
        Initialize the navigator.
        
        Args:
            pattern: The navigation pattern
            phi_weights: Dict of (signs, exponents) for each weight
            encoder: PhiEncoder for decoding weights
            device: Compute device
        """
        self.pattern = pattern
        self.phi_weights = phi_weights
        self.encoder = encoder
        self.device = device
        
        # Decode weights for computation
        self.weights = {}
        for name, (signs, exps) in phi_weights.items():
            self.weights[name] = encoder.decode(signs, exps).to(device)
    
    def navigate(
        self,
        input_tensor: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Navigate through the shape to produce output.
        
        Args:
            input_tensor: Input data
            context: Optional context for cross-attention
            
        Returns:
            Output tensor
        """
        x = input_tensor.float().to(self.device)
        
        # Ensure at least 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Navigate according to pattern topology
        if self.pattern.topology == Topology.CONVERGENT:
            return self._navigate_funnel(x)
        elif self.pattern.topology == Topology.SPIRAL:
            return self._navigate_spiral(x)
        elif self.pattern.topology == Topology.WEB:
            return self._navigate_web(x, context)
        elif self.pattern.topology == Topology.DIVERGENT:
            return self._navigate_tree(x)
        elif self.pattern.topology == Topology.BRAID:
            return self._navigate_braid(x, context)
        elif self.pattern.topology == Topology.HOURGLASS:
            return self._navigate_hourglass(x)
        elif self.pattern.topology == Topology.RING:
            return self._navigate_ring(x)
        else:
            return self._navigate_generic(x)
    
    def _get_weight(self, name: str) -> Optional[torch.Tensor]:
        """Get a weight tensor by name."""
        return self.weights.get(name)
    
    def _linear(self, x: torch.Tensor, weight_name: str) -> torch.Tensor:
        """Apply linear transformation."""
        W = self._get_weight(f"{weight_name}.weight")
        b = self._get_weight(f"{weight_name}.bias")
        
        if W is None:
            return x
        
        # Handle dimension mismatch
        if x.shape[-1] != W.shape[1]:
            if x.shape[-1] < W.shape[1]:
                padding = torch.zeros(*x.shape[:-1], W.shape[1] - x.shape[-1], device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[..., :W.shape[1]]
        
        out = x @ W.T
        if b is not None:
            out = out + b
        return out
    
    def _self_attention(self, x: torch.Tensor, node_name: str) -> torch.Tensor:
        """Apply self-attention."""
        W = self._get_weight(f"{node_name}.weight")
        if W is None:
            return x
        
        # Simplified attention: x @ W @ x.T
        # In full implementation, would use Q, K, V projections
        if x.shape[-1] != W.shape[1]:
            if x.shape[-1] < W.shape[1]:
                padding = torch.zeros(*x.shape[:-1], W.shape[1] - x.shape[-1], device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[..., :W.shape[1]]
        
        # Compute attention scores
        scores = x @ W @ x.T
        
        # Softmax
        attn = F.softmax(scores / (x.shape[-1] ** 0.5), dim=-1)
        
        # Apply attention
        return attn @ x
    
    def _cross_attention(
        self, 
        x: torch.Tensor, 
        context: torch.Tensor, 
        node_name: str
    ) -> torch.Tensor:
        """Apply cross-attention."""
        W = self._get_weight(f"{node_name}.weight")
        if W is None:
            return x
        
        # Simplified cross-attention
        if context is None:
            context = x
        
        # Ensure dimensions match
        if x.shape[-1] != W.shape[1]:
            if x.shape[-1] < W.shape[1]:
                padding = torch.zeros(*x.shape[:-1], W.shape[1] - x.shape[-1], device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[..., :W.shape[1]]
        
        if context.shape[-1] != W.shape[0]:
            if context.shape[-1] < W.shape[0]:
                padding = torch.zeros(*context.shape[:-1], W.shape[0] - context.shape[-1], device=context.device)
                context = torch.cat([context, padding], dim=-1)
            else:
                context = context[..., :W.shape[0]]
        
        # Compute cross-attention
        scores = x @ W @ context.T
        attn = F.softmax(scores / (x.shape[-1] ** 0.5), dim=-1)
        
        return attn @ context
    
    def _ffn(self, x: torch.Tensor, node_name: str) -> torch.Tensor:
        """Apply feed-forward network."""
        W = self._get_weight(f"{node_name}.weight")
        if W is None:
            return x
        
        # Handle dimension mismatch
        if x.shape[-1] != W.shape[1]:
            if x.shape[-1] < W.shape[1]:
                padding = torch.zeros(*x.shape[:-1], W.shape[1] - x.shape[-1], device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[..., :W.shape[1]]
        
        # FFN with ReLU
        return F.relu(x @ W.T)
    
    def _navigate_funnel(self, x: torch.Tensor) -> torch.Tensor:
        """Navigate convergent (funnel) pattern."""
        for node in self.pattern.nodes:
            x = self._linear(x, node.name)
            if 'compress' in node.name:
                x = F.relu(x)
        return x
    
    def _navigate_spiral(self, x: torch.Tensor) -> torch.Tensor:
        """Navigate self-referential (spiral) pattern."""
        for node in self.pattern.nodes:
            if node.node_type == "self_attention":
                x = x + self._self_attention(x, node.name)  # Residual
            elif node.node_type == "linear":
                if "up" in node.name:
                    x = F.relu(self._linear(x, node.name))
                else:
                    x = x + self._linear(x, node.name)  # Residual
        return x
    
    def _navigate_web(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Navigate cross-connected (web) pattern."""
        if context is None:
            context = x
        
        for node in self.pattern.nodes:
            if node.node_type == "cross_attention":
                x = x + self._cross_attention(x, context, node.name)
            elif node.node_type == "self_attention":
                x = x + self._self_attention(x, node.name)
            elif node.node_type == "ffn":
                x = x + self._ffn(x, node.name)
            elif node.node_type == "linear":
                x = self._linear(x, node.name)
        return x
    
    def _navigate_tree(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Navigate divergent (tree) pattern."""
        outputs = {}
        for node in self.pattern.nodes:
            branch_name = node.name.replace("branch_", "")
            outputs[branch_name] = self._linear(x, node.name)
        return outputs
    
    def _navigate_braid(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Navigate intertwined (braid) pattern."""
        # For braid, x should be a dict of stream tensors
        # Simplified: treat as single stream
        for node in self.pattern.nodes:
            if node.node_type == "self_attention":
                x = x + self._self_attention(x, node.name)
            elif node.node_type == "cross_attention":
                x = x + self._cross_attention(x, context or x, node.name)
            elif node.node_type == "ffn":
                x = x + self._ffn(x, node.name)
        return x
    
    def _navigate_hourglass(self, x: torch.Tensor) -> torch.Tensor:
        """Navigate compress/expand (hourglass) pattern."""
        skip_connections = {}
        
        for node in self.pattern.nodes:
            if "encode" in node.name:
                skip_connections[node.name] = x
                x = F.relu(self._linear(x, node.name))
            elif "decode" in node.name:
                x = self._linear(x, node.name)
                # Add skip connection if available
                skip_name = node.params.get("skip_from")
                if skip_name and skip_name in skip_connections:
                    skip = skip_connections[skip_name]
                    if skip.shape == x.shape:
                        x = x + skip
        return x
    
    def _navigate_ring(self, x: torch.Tensor) -> torch.Tensor:
        """Navigate closed loop (ring) pattern."""
        # Ring requires state management
        # Simplified: single pass
        for node in self.pattern.nodes:
            x = self._linear(x, node.name)
            if "state" in node.name:
                x = torch.tanh(x)
        return x
    
    def _navigate_generic(self, x: torch.Tensor) -> torch.Tensor:
        """Generic navigation for any pattern."""
        for node in self.pattern.nodes:
            if node.node_type == "linear":
                x = self._linear(x, node.name)
            elif node.node_type == "self_attention":
                x = self._self_attention(x, node.name)
            elif node.node_type == "cross_attention":
                x = self._cross_attention(x, x, node.name)
            elif node.node_type == "ffn":
                x = self._ffn(x, node.name)
        return x


def test_navigator():
    """Test the navigator."""
    print("=" * 60)
    print("NAVIGATOR TEST")
    print("=" * 60)
    
    from .patterns import Funnel, Spiral, Web
    from .projector import ShapeProjector, ProblemSpec, IOSpec, DataType
    
    encoder = PhiEncoder()
    projector = ShapeProjector(encoder)
    
    # Test Funnel
    print("\n1. Funnel navigation:")
    problem = ProblemSpec(
        name="test",
        inputs=[IOSpec("x", DataType.VECTOR, (64,))],
        outputs=[IOSpec("y", DataType.VECTOR, (10,))],
    )
    pattern, weights = projector.project(problem)
    navigator = Navigator(pattern, weights, encoder)
    
    x = torch.randn(64)
    y = navigator.navigate(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {y.shape}")
    
    # Test Spiral
    print("\n2. Spiral navigation:")
    problem = ProblemSpec(
        name="test",
        inputs=[IOSpec("x", DataType.SEQUENCE, (32,))],
        outputs=[IOSpec("y", DataType.VECTOR, (50,))],
        temporal=True,
    )
    pattern, weights = projector.project(problem)
    navigator = Navigator(pattern, weights, encoder)
    
    x = torch.randn(32)
    y = navigator.navigate(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {y.shape}")
    
    # Test Web
    print("\n3. Web navigation:")
    problem = ProblemSpec(
        name="test",
        inputs=[IOSpec("x", DataType.IMAGE, (8, 8, 1))],
        outputs=[IOSpec("y", DataType.IMAGE, (8, 8, 2))],
    )
    pattern, weights = projector.project(problem)
    navigator = Navigator(pattern, weights, encoder)
    
    x = torch.randn(64)  # Flattened
    y = navigator.navigate(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {y.shape}")
    
    print("\n" + "=" * 60)
    print("NAVIGATOR TEST COMPLETE")
    print("=" * 60)
    
    return navigator


if __name__ == "__main__":
    test_navigator()
