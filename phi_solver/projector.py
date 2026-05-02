#!/usr/bin/env python3
"""
Shape Projector: Derive φ-coordinates from problem structure.

The key insight: The problem itself defines the shape.
- Input/output dimensions define the funnel width
- Temporal dependencies define spiral depth
- Cross-modal relationships define web connectivity

This is geometric AI design from first principles.

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

from .encoder import PhiEncoder, PHI, LN_PHI
from .pattern import (
    Pattern, Topology, SelfReference, PatternNode,
    Funnel, Spiral, Web, Tree, Hourglass, Braid
)

# =============================================================================
# PROBLEM SPECIFICATION
# =============================================================================

class DataType(Enum):
    """Types of data that can be input/output."""
    SCALAR = "scalar"           # Single number
    VECTOR = "vector"           # 1D array
    IMAGE = "image"             # 2D spatial (H, W, C)
    SEQUENCE = "sequence"       # 1D temporal (T, D)
    GRAPH = "graph"             # Nodes + edges
    SET = "set"                 # Unordered collection


@dataclass
class IOSpec:
    """Specification of an input or output."""
    name: str
    data_type: DataType
    shape: Tuple[int, ...]      # Dimensions (excluding batch)
    semantic: str = ""          # What it represents ("color", "depth", etc.)
    
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
    """Full specification of a problem."""
    name: str
    inputs: List[IOSpec]
    outputs: List[IOSpec]
    
    # Relationships
    temporal: bool = False      # Does order matter?
    hierarchical: bool = False  # Multiple scales?
    cross_modal: bool = False   # Different modalities interact?
    
    # Constraints
    causal: bool = False        # Can only see past?
    symmetric: bool = False     # Input/output same structure?
    
    def describe(self) -> str:
        in_str = ", ".join(f"{i.name}:{i.data_type.value}{i.shape}" for i in self.inputs)
        out_str = ", ".join(f"{o.name}:{o.data_type.value}{o.shape}" for o in self.outputs)
        return f"{self.name}: [{in_str}] → [{out_str}]"


# =============================================================================
# SHAPE PROJECTOR
# =============================================================================

class ShapeProjector:
    """
    Project φ-shapes from problem specifications.
    
    The projector analyzes the problem structure and derives:
    1. The appropriate pattern (Funnel, Spiral, Web, etc.)
    2. The dimensions and depth
    3. Initial φ-coordinates for weights
    
    This is geometric AI design - no training required.
    """
    
    def __init__(self, encoder: Optional[PhiEncoder] = None):
        self.encoder = encoder or PhiEncoder(K=32)
    
    def project(self, problem: ProblemSpec) -> Tuple[Pattern, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Project a φ-shape from a problem specification.
        
        Args:
            problem: The problem specification
            
        Returns:
            pattern: The navigation pattern
            phi_weights: Dict of (signs, exponents) for initial weights
        """
        # Step 1: Determine the pattern topology
        pattern = self._select_pattern(problem)
        
        # Step 2: Compute dimensions
        dims = self._compute_dimensions(problem, pattern)
        
        # Step 3: Project initial φ-coordinates
        phi_weights = self._project_coordinates(problem, pattern, dims)
        
        return pattern, phi_weights
    
    def _select_pattern(self, problem: ProblemSpec) -> Pattern:
        """Select the appropriate pattern based on problem structure."""
        
        total_in = sum(i.total_dim for i in problem.inputs)
        total_out = sum(o.total_dim for o in problem.outputs)
        
        # Decision tree for pattern selection
        
        # Multiple outputs from single input → Tree
        if len(problem.outputs) > 1 and len(problem.inputs) == 1:
            branches = [(o.name, o.total_dim) for o in problem.outputs]
            return Tree(in_dim=total_in, branches=branches)
        
        # Cross-modal (different input types) → Braid or Web
        if problem.cross_modal or len(problem.inputs) > 1:
            input_types = set(i.data_type for i in problem.inputs)
            if len(input_types) > 1:
                # Different modalities → Braid
                streams = [i.name for i in problem.inputs]
                dim = max(i.total_dim for i in problem.inputs)
                return Braid(streams=streams, dim=dim, layers=6, cross_every=2)
            else:
                # Same modality, cross-attention → Web
                queries = min(100, total_out)
                return Web(queries=queries, dim=256, feature_scales=3, 
                          layers=9, output_dim=total_out)
        
        # Temporal/sequential → Spiral
        if problem.temporal or any(i.is_temporal for i in problem.inputs):
            depth = self._estimate_depth(problem)
            dim = max(total_in, total_out, 256)
            heads = max(1, dim // 64)
            return Spiral(layers=depth, dim=dim, heads=heads)
        
        # Symmetric (autoencoder-like) → Hourglass
        if problem.symmetric:
            bottleneck = min(total_in, total_out) // 4
            dims = [total_in, total_in // 2, total_in // 4]
            return Hourglass(dims=dims, bottleneck_dim=bottleneck)
        
        # Spatial with different output → Web (like colorization)
        if any(i.is_spatial for i in problem.inputs) and total_out != total_in:
            queries = min(100, max(total_out, 32))
            return Web(queries=queries, dim=256, feature_scales=3,
                      layers=9, output_dim=total_out)
        
        # Default: Funnel (simple prediction)
        hidden = max(32, min(total_in // 4, 256))
        return Funnel(in_dim=total_in, out_dim=total_out, hidden_dim=hidden)
    
    def _estimate_depth(self, problem: ProblemSpec) -> int:
        """Estimate the depth needed for a problem."""
        total_in = sum(i.total_dim for i in problem.inputs)
        total_out = sum(o.total_dim for o in problem.outputs)
        
        # Heuristic: more complex problems need more depth
        complexity = np.log2(max(total_in, total_out, 2))
        
        if problem.hierarchical:
            complexity *= 1.5
        if problem.causal:
            complexity *= 1.2
        
        return max(4, min(32, int(complexity * 2)))
    
    def _compute_dimensions(self, problem: ProblemSpec, pattern: Pattern) -> Dict[str, int]:
        """Compute the dimensions for each layer."""
        total_in = sum(i.total_dim for i in problem.inputs)
        total_out = sum(o.total_dim for o in problem.outputs)
        
        dims = {
            'input': total_in,
            'output': total_out,
            'hidden': max(64, (total_in + total_out) // 2)
        }
        
        # Pattern-specific dimensions
        if isinstance(pattern, Spiral):
            dims['hidden'] = pattern.dim
            dims['heads'] = pattern.heads
            dims['layers'] = pattern.layers
        elif isinstance(pattern, Web):
            dims['queries'] = pattern.queries
            dims['hidden'] = pattern.dim
        elif isinstance(pattern, Funnel):
            dims['hidden'] = pattern.nodes[0].out_dim if len(pattern.nodes) > 1 else total_out
        
        return dims
    
    def _project_coordinates(
        self, 
        problem: ProblemSpec, 
        pattern: Pattern, 
        dims: Dict[str, int]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Project initial φ-coordinates for the weights.
        
        This is the key innovation: deriving coordinates from structure.
        """
        phi_weights = {}
        
        for node in pattern.nodes:
            weight_name = f"{node.name}.weight"
            
            # Create weight matrix
            W = self._create_geometric_weight(
                node.in_dim, 
                node.out_dim, 
                node.node_type,
                problem
            )
            
            # Encode in φ-basis
            signs, exps = self.encoder.encode(W)
            phi_weights[weight_name] = (signs, exps)
            
            # Bias if needed
            if node.node_type in ['linear', 'ffn']:
                bias = self._create_geometric_bias(node.out_dim, problem)
                b_signs, b_exps = self.encoder.encode(bias)
                phi_weights[f"{node.name}.bias"] = (b_signs, b_exps)
        
        return phi_weights
    
    def _create_geometric_weight(
        self, 
        in_dim: int, 
        out_dim: int, 
        node_type: str,
        problem: ProblemSpec
    ) -> torch.Tensor:
        """
        Create a weight matrix using geometric principles.
        
        Key insight: The weight should encode the RELATIONSHIP between
        input and output spaces. We use φ-based construction.
        """
        W = torch.zeros(out_dim, in_dim)
        
        if node_type == 'linear':
            # Linear projection: φ-orthogonal basis
            W = self._phi_orthogonal_projection(in_dim, out_dim)
            
        elif node_type == 'self_attention':
            # Self-attention: φ-rotation matrices
            W = self._phi_attention_weight(in_dim, out_dim)
            
        elif node_type == 'cross_attention':
            # Cross-attention: φ-bridge between spaces
            W = self._phi_cross_attention_weight(in_dim, out_dim)
            
        elif node_type == 'ffn':
            # FFN: φ-expansion/contraction
            W = self._phi_ffn_weight(in_dim, out_dim)
            
        else:
            # Default: scaled φ-random
            W = self._phi_random_weight(in_dim, out_dim)
        
        return W
    
    def _phi_orthogonal_projection(self, in_dim: int, out_dim: int) -> torch.Tensor:
        """Create a φ-orthogonal projection matrix (vectorized)."""
        # Simple φ-scaled random - fast and effective
        exponents = torch.randn(out_dim, in_dim) * 2 - 9
        signs = torch.sign(torch.randn(out_dim, in_dim))
        signs[signs == 0] = 1
        
        W = signs * (PHI ** exponents)
        
        # Scale rows by φ-levels
        levels = (torch.arange(out_dim).float() - out_dim // 2) / 10
        W = W * (PHI ** levels).unsqueeze(1)
        
        return W
    
    def _phi_attention_weight(self, in_dim: int, out_dim: int) -> torch.Tensor:
        """Create attention weight using φ-rotations (vectorized)."""
        # Diagonal with φ-scaling
        diag_size = min(out_dim, in_dim)
        levels = (torch.arange(diag_size).float() - diag_size // 2) / 20
        diag_vals = PHI ** levels
        
        W = torch.zeros(out_dim, in_dim)
        W[:diag_size, :diag_size] = torch.diag(diag_vals)
        
        # Add local φ-connections (band matrix)
        for offset in range(1, 6):
            val = (PHI ** (-offset)) * 0.1
            if offset < out_dim and offset < in_dim:
                W.diagonal(offset)[:] = val
                W.diagonal(-offset)[:] = val
        
        return W
    
    def _phi_cross_attention_weight(self, in_dim: int, out_dim: int) -> torch.Tensor:
        """Create cross-attention weight bridging two spaces (vectorized)."""
        # Create sparse φ-bridge using broadcasting
        out_idx = torch.arange(out_dim).float()
        centers = (out_idx / out_dim * in_dim).long()
        
        # Each row has 7 connections centered at its mapped position
        offsets = torch.arange(-3, 4)
        weights = PHI ** (-torch.abs(offsets).float())
        
        W = torch.zeros(out_dim, in_dim)
        for k, offset in enumerate(offsets):
            cols = (centers + offset) % in_dim
            W[torch.arange(out_dim), cols] = weights[k]
        
        # Normalize rows
        W = W / (W.norm(dim=1, keepdim=True) + 1e-10)
        
        return W
    
    def _phi_ffn_weight(self, in_dim: int, out_dim: int) -> torch.Tensor:
        """Create FFN weight using φ-expansion (vectorized)."""
        # Simple φ-scaled random initialization
        # Much faster than sparse construction
        exponents = torch.randn(out_dim, in_dim) * 2 - 9
        signs = torch.sign(torch.randn(out_dim, in_dim))
        signs[signs == 0] = 1
        
        W = signs * (PHI ** exponents)
        return W
    
    def _phi_random_weight(self, in_dim: int, out_dim: int) -> torch.Tensor:
        """Create random weight on φ-lattice."""
        # Random exponents centered around φ^-9 (typical weight magnitude)
        exponents = torch.randn(out_dim, in_dim) * 3 - 9
        signs = torch.sign(torch.randn(out_dim, in_dim))
        signs[signs == 0] = 1
        
        W = signs * (PHI ** exponents)
        return W
    
    def _create_geometric_bias(self, dim: int, problem: ProblemSpec) -> torch.Tensor:
        """Create bias using geometric principles."""
        # Biases are typically small, centered around 0
        # Use φ-levels near 0
        exponents = torch.randn(dim) * 2 - 10  # Small values
        signs = torch.sign(torch.randn(dim))
        signs[signs == 0] = 1
        
        return signs * (PHI ** exponents)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def project_colorization() -> Tuple[Pattern, Dict]:
    """Project a shape for image colorization."""
    problem = ProblemSpec(
        name="colorization",
        inputs=[IOSpec("grayscale", DataType.IMAGE, (512, 512, 1), "luminance")],
        outputs=[IOSpec("color", DataType.IMAGE, (512, 512, 2), "ab channels")],
        cross_modal=False,
        hierarchical=True
    )
    projector = ShapeProjector()
    return projector.project(problem)


def project_depth() -> Tuple[Pattern, Dict]:
    """Project a shape for depth estimation."""
    problem = ProblemSpec(
        name="depth_estimation",
        inputs=[IOSpec("rgb", DataType.IMAGE, (512, 512, 3), "appearance")],
        outputs=[IOSpec("depth", DataType.IMAGE, (512, 512, 1), "geometry")],
        hierarchical=True
    )
    projector = ShapeProjector()
    return projector.project(problem)


def project_classification(num_classes: int = 1000) -> Tuple[Pattern, Dict]:
    """Project a shape for image classification."""
    problem = ProblemSpec(
        name="classification",
        inputs=[IOSpec("image", DataType.IMAGE, (224, 224, 3), "visual")],
        outputs=[IOSpec("class", DataType.VECTOR, (num_classes,), "category")],
    )
    projector = ShapeProjector()
    return projector.project(problem)


def project_language_model(vocab_size: int = 32000, context: int = 2048) -> Tuple[Pattern, Dict]:
    """Project a shape for language modeling."""
    problem = ProblemSpec(
        name="language_model",
        inputs=[IOSpec("tokens", DataType.SEQUENCE, (context,), "text")],
        outputs=[IOSpec("next_token", DataType.VECTOR, (vocab_size,), "prediction")],
        temporal=True,
        causal=True
    )
    projector = ShapeProjector()
    return projector.project(problem)


def project_multimodal(image_dim: int = 512, text_dim: int = 256) -> Tuple[Pattern, Dict]:
    """Project a shape for vision-language tasks."""
    problem = ProblemSpec(
        name="vision_language",
        inputs=[
            IOSpec("image", DataType.IMAGE, (224, 224, 3), "visual"),
            IOSpec("text", DataType.SEQUENCE, (128,), "language")
        ],
        outputs=[IOSpec("answer", DataType.VECTOR, (text_dim,), "response")],
        cross_modal=True,
        temporal=True
    )
    projector = ShapeProjector()
    return projector.project(problem)


# =============================================================================
# TEST
# =============================================================================

def test_projector():
    """Test the shape projector."""
    print("=" * 70)
    print("SHAPE PROJECTOR TEST")
    print("=" * 70)
    
    projector = ShapeProjector()
    
    # Test various problems
    problems = [
        ("Colorization", project_colorization),
        ("Depth Estimation", project_depth),
        ("Classification", lambda: project_classification(1000)),
        ("Language Model", lambda: project_language_model(32000, 512)),
        ("Vision-Language", project_multimodal),
    ]
    
    for name, project_fn in problems:
        print(f"\n{name}:")
        pattern, phi_weights = project_fn()
        print(f"  Pattern: {pattern.name} ({pattern.topology.value})")
        print(f"  Nodes: {len(pattern.nodes)}")
        print(f"  Weights: {len(phi_weights)} tensors")
        
        # Show weight shapes
        total_params = 0
        for w_name, (signs, exps) in phi_weights.items():
            params = signs.numel()
            total_params += params
            if params > 1000:
                print(f"    {w_name}: {list(signs.shape)} ({params:,} params)")
        print(f"  Total params: {total_params:,}")
    
    return projector


if __name__ == "__main__":
    test_projector()
