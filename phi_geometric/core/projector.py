"""
Shape Projector: Derive φ-coordinates from problem structure.

The key insight: The problem itself defines the shape.
    - Input/output dimensions define the funnel width
    - Temporal dependencies define spiral depth
    - Cross-modal relationships define web connectivity

This is geometric AI design from first principles - no training required.

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from .encoder import PhiEncoder, PHI, LN_PHI
from .patterns import (
    Pattern, Topology, SelfReference,
    Funnel, Spiral, Web, Tree, Braid, Hourglass
)


class DataType(Enum):
    """Types of data that can be input/output."""
    SCALAR = "scalar"       # Single number
    VECTOR = "vector"       # 1D array
    IMAGE = "image"         # 2D spatial (H, W, C)
    SEQUENCE = "sequence"   # 1D temporal (T, D)
    GRAPH = "graph"         # Nodes + edges
    SET = "set"             # Unordered collection


@dataclass
class IOSpec:
    """
    Specification of an input or output.
    
    Attributes:
        name: Identifier for this I/O
        data_type: Type of data (scalar, vector, image, etc.)
        shape: Dimensions (excluding batch)
        semantic: What it represents ("color", "depth", etc.)
    """
    name: str
    data_type: DataType
    shape: Tuple[int, ...]
    semantic: str = ""
    
    @property
    def total_dim(self) -> int:
        """Total dimensionality."""
        return int(np.prod(self.shape))
    
    @property
    def is_spatial(self) -> bool:
        return self.data_type == DataType.IMAGE
    
    @property
    def is_temporal(self) -> bool:
        return self.data_type == DataType.SEQUENCE


@dataclass
class ProblemSpec:
    """
    Full specification of a problem.
    
    Attributes:
        name: Problem identifier
        inputs: List of input specifications
        outputs: List of output specifications
        temporal: Does order matter?
        hierarchical: Multiple scales?
        cross_modal: Different modalities interact?
        causal: Can only see past?
        symmetric: Input/output same structure?
    """
    name: str
    inputs: List[IOSpec]
    outputs: List[IOSpec]
    temporal: bool = False
    hierarchical: bool = False
    cross_modal: bool = False
    causal: bool = False
    symmetric: bool = False
    
    def describe(self) -> str:
        in_str = ", ".join(f"{i.name}:{i.data_type.value}" for i in self.inputs)
        out_str = ", ".join(f"{o.name}:{o.data_type.value}" for o in self.outputs)
        return f"{self.name}: [{in_str}] → [{out_str}]"


class ShapeProjector:
    """
    Project φ-shapes from problem specifications.
    
    The projector analyzes the problem structure and derives:
        1. The appropriate pattern (Funnel, Spiral, Web, etc.)
        2. The dimensions and depth
        3. Initial φ-coordinates for weights
    
    This is geometric AI design - no training required.
    
    Example:
        projector = ShapeProjector()
        
        problem = ProblemSpec(
            name="colorization",
            inputs=[IOSpec("gray", DataType.IMAGE, (512, 512, 1))],
            outputs=[IOSpec("color", DataType.IMAGE, (512, 512, 2))],
        )
        
        pattern, phi_weights = projector.project(problem)
        # pattern = Web, phi_weights = {name: (signs, exponents)}
    """
    
    def __init__(self, encoder: Optional[PhiEncoder] = None):
        self.encoder = encoder or PhiEncoder(K=32)
    
    def project(
        self, 
        problem: ProblemSpec
    ) -> Tuple[Pattern, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Project a φ-shape from a problem specification.
        
        Args:
            problem: The problem specification
            
        Returns:
            pattern: The navigation pattern
            phi_weights: Dict of (signs, exponents) for initial weights
        """
        # Step 1: Select pattern
        pattern = self._select_pattern(problem)
        
        # Step 2: Compute dimensions
        dims = self._compute_dimensions(problem, pattern)
        
        # Step 3: Project φ-coordinates
        phi_weights = self._project_coordinates(problem, pattern, dims)
        
        return pattern, phi_weights
    
    def _select_pattern(self, problem: ProblemSpec) -> Pattern:
        """Select the appropriate pattern based on problem structure."""
        total_in = sum(i.total_dim for i in problem.inputs)
        total_out = sum(o.total_dim for o in problem.outputs)
        
        # Multiple outputs from single input → Tree
        if len(problem.outputs) > 1 and len(problem.inputs) == 1:
            branches = [(o.name, o.total_dim) for o in problem.outputs]
            return Tree(in_dim=total_in, branches=branches)
        
        # Cross-modal or multiple input types → Braid or Web
        if problem.cross_modal or len(problem.inputs) > 1:
            input_types = set(i.data_type for i in problem.inputs)
            if len(input_types) > 1:
                streams = [i.name for i in problem.inputs]
                dim = max(i.total_dim for i in problem.inputs)
                dim = min(dim, 512)  # Cap dimension
                return Braid(streams=streams, dim=dim, layers=6, cross_every=2)
            else:
                queries = min(100, total_out)
                return Web(queries=queries, dim=256, feature_scales=3,
                          layers=9, output_dim=total_out)
        
        # Temporal/sequential → Spiral
        if problem.temporal or any(i.is_temporal for i in problem.inputs):
            depth = self._estimate_depth(problem)
            dim = max(total_in, total_out, 256)
            dim = min(dim, 1024)  # Cap dimension
            heads = max(1, dim // 64)
            return Spiral(layers=depth, dim=dim, heads=heads)
        
        # Symmetric → Hourglass
        if problem.symmetric:
            bottleneck = min(total_in, total_out) // 4
            dims = [total_in, total_in // 2, total_in // 4]
            return Hourglass(dims=dims, bottleneck_dim=max(16, bottleneck))
        
        # Spatial with different output → Web
        if any(i.is_spatial for i in problem.inputs) and total_out != total_in:
            queries = min(100, max(total_out, 32))
            return Web(queries=queries, dim=256, feature_scales=3,
                      layers=9, output_dim=total_out)
        
        # Default: Funnel
        hidden = max(32, min(total_in // 4, 256))
        return Funnel(in_dim=total_in, out_dim=total_out, hidden_dim=hidden)
    
    def _estimate_depth(self, problem: ProblemSpec) -> int:
        """Estimate the depth needed for a problem."""
        total_in = sum(i.total_dim for i in problem.inputs)
        total_out = sum(o.total_dim for o in problem.outputs)
        
        complexity = np.log2(max(total_in, total_out, 2))
        
        if problem.hierarchical:
            complexity *= 1.5
        if problem.causal:
            complexity *= 1.2
        
        return max(4, min(32, int(complexity * 2)))
    
    def _compute_dimensions(self, problem: ProblemSpec, pattern: Pattern) -> Dict[str, int]:
        """Compute dimensions for each layer."""
        total_in = sum(i.total_dim for i in problem.inputs)
        total_out = sum(o.total_dim for o in problem.outputs)
        
        return {
            'input': total_in,
            'output': total_out,
            'hidden': max(64, (total_in + total_out) // 2)
        }
    
    def _project_coordinates(
        self,
        problem: ProblemSpec,
        pattern: Pattern,
        dims: Dict[str, int]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Project initial φ-coordinates for weights."""
        phi_weights = {}
        
        for node in pattern.nodes:
            weight_name = f"{node.name}.weight"
            
            # Create weight matrix
            W = self._create_phi_weight(node.in_dim, node.out_dim, node.node_type)
            
            # Encode in φ-basis
            signs, exps = self.encoder.encode(W)
            phi_weights[weight_name] = (signs, exps)
            
            # Bias for linear/ffn layers
            if node.node_type in ['linear', 'ffn']:
                bias = self._create_phi_bias(node.out_dim)
                b_signs, b_exps = self.encoder.encode(bias)
                phi_weights[f"{node.name}.bias"] = (b_signs, b_exps)
        
        return phi_weights
    
    def _create_phi_weight(self, in_dim: int, out_dim: int, node_type: str) -> torch.Tensor:
        """Create a weight matrix using φ-geometric principles."""
        # φ-scaled random initialization
        # Exponents centered around φ^-9 (typical weight magnitude)
        exponents = torch.randn(out_dim, in_dim) * 2 - 9
        signs = torch.sign(torch.randn(out_dim, in_dim))
        signs[signs == 0] = 1
        
        W = signs * (PHI ** exponents)
        
        # Scale rows by φ-levels for structure
        if node_type in ['self_attention', 'cross_attention']:
            levels = (torch.arange(out_dim).float() - out_dim // 2) / 20
            W = W * (PHI ** levels).unsqueeze(1)
        
        return W
    
    def _create_phi_bias(self, dim: int) -> torch.Tensor:
        """Create bias using φ-geometric principles."""
        exponents = torch.randn(dim) * 2 - 10  # Small values
        signs = torch.sign(torch.randn(dim))
        signs[signs == 0] = 1
        return signs * (PHI ** exponents)


def test_projector():
    """Test the shape projector."""
    print("=" * 60)
    print("SHAPE PROJECTOR TEST")
    print("=" * 60)
    
    projector = ShapeProjector()
    
    problems = [
        ProblemSpec(
            name="classifier",
            inputs=[IOSpec("x", DataType.VECTOR, (64,))],
            outputs=[IOSpec("y", DataType.VECTOR, (10,))],
        ),
        ProblemSpec(
            name="colorizer",
            inputs=[IOSpec("gray", DataType.IMAGE, (32, 32, 1))],
            outputs=[IOSpec("color", DataType.IMAGE, (32, 32, 2))],
        ),
        ProblemSpec(
            name="language",
            inputs=[IOSpec("tokens", DataType.SEQUENCE, (64,))],
            outputs=[IOSpec("next", DataType.VECTOR, (100,))],
            temporal=True,
        ),
        ProblemSpec(
            name="multimodal",
            inputs=[
                IOSpec("image", DataType.IMAGE, (32, 32, 3)),
                IOSpec("text", DataType.SEQUENCE, (16,))
            ],
            outputs=[IOSpec("answer", DataType.VECTOR, (64,))],
            cross_modal=True,
        ),
    ]
    
    for problem in problems:
        pattern, weights = projector.project(problem)
        total_params = sum(s.numel() for s, e in weights.values())
        print(f"\n{problem.name}:")
        print(f"  Pattern: {pattern.name}")
        print(f"  Nodes: {len(pattern.nodes)}")
        print(f"  Params: {total_params:,}")
    
    print("\n" + "=" * 60)
    print("SHAPE PROJECTOR TEST COMPLETE")
    print("=" * 60)
    
    return projector


if __name__ == "__main__":
    test_projector()
