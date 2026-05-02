"""
Pattern: Define the topology of navigation through the φ-lattice.

Patterns describe HOW information flows through a model:
- Funnel: Convergent (many → one)
- Spiral: Self-referential (deep with attention)
- Web: Cross-connected (queries attend to features)
- Tree: Divergent (one → many)
- Braid: Intertwined streams
- Hourglass: Compress then expand
- Ring: Closed loop (recurrent)
- Constellation: Graph-structured
- Fractal: Self-similar at scales
- Mirror: Symmetric translation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Union
from enum import Enum


class Topology(Enum):
    """The fundamental shape of information flow."""
    CONVERGENT = "convergent"      # Many → One (Funnel)
    DIVERGENT = "divergent"        # One → Many (Tree)
    SPIRAL = "spiral"              # Self-referential loop (Transformer)
    WEB = "web"                    # Cross-connected mesh
    BRAID = "braid"                # Intertwined parallel streams
    HOURGLASS = "hourglass"        # Compress → Expand
    RING = "ring"                  # Closed recurrent loop
    CONSTELLATION = "constellation" # Graph structure
    FRACTAL = "fractal"            # Self-similar hierarchy
    MIRROR = "mirror"              # Symmetric translation


class SelfReference(Enum):
    """How much the pattern references itself."""
    NONE = "none"          # Pure feedforward
    PARTIAL = "partial"    # Some self-attention
    FULL = "full"          # Every layer has self-attention
    RECURRENT = "recurrent" # Output feeds back as input


@dataclass
class PatternNode:
    """A node in the pattern graph."""
    name: str
    node_type: str  # "linear", "attention", "cross_attention", "ffn", etc.
    in_dim: int
    out_dim: int
    params: Dict = field(default_factory=dict)


@dataclass
class Pattern:
    """
    A pattern defines the topology of navigation through the φ-lattice.
    
    Attributes:
        name: Human-readable name
        topology: The fundamental shape (Funnel, Spiral, Web, etc.)
        self_reference: How much the pattern references itself
        io_ratio: Input:Output ratio ("N:1", "1:N", "N:M", "1:1")
        nodes: List of nodes in the pattern graph
    """
    name: str
    topology: Topology
    self_reference: SelfReference = SelfReference.NONE
    io_ratio: str = "N:M"
    nodes: List[PatternNode] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def add_node(self, node: PatternNode):
        """Add a node to the pattern."""
        self.nodes.append(node)
    
    def describe(self) -> str:
        """Human-readable description of the pattern."""
        return (
            f"Pattern: {self.name}\n"
            f"  Topology: {self.topology.value}\n"
            f"  Self-reference: {self.self_reference.value}\n"
            f"  I/O ratio: {self.io_ratio}\n"
            f"  Nodes: {len(self.nodes)}"
        )


# =============================================================================
# PRESET PATTERNS
# =============================================================================

class Funnel(Pattern):
    """
    Convergent pattern: many inputs → one output.
    
    Used for: Depth estimation, classification, regression
    Example: DA2 head (1024 → 32 → 1)
    """
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__(
            name="funnel",
            topology=Topology.CONVERGENT,
            self_reference=SelfReference.NONE,
            io_ratio="N:1"
        )
        
        if hidden_dim:
            self.add_node(PatternNode("compress", "linear", in_dim, hidden_dim))
            self.add_node(PatternNode("output", "linear", hidden_dim, out_dim))
        else:
            self.add_node(PatternNode("output", "linear", in_dim, out_dim))
        
        self.in_dim = in_dim
        self.out_dim = out_dim


class Spiral(Pattern):
    """
    Self-referential pattern: deep with attention at every layer.
    
    Used for: Language modeling, reasoning
    Example: Qwen2-7B (28 layers of self-attention + FFN)
    """
    
    def __init__(self, layers: int, dim: int, heads: int, ffn_dim: Optional[int] = None):
        super().__init__(
            name="spiral",
            topology=Topology.SPIRAL,
            self_reference=SelfReference.FULL,
            io_ratio="1:1"
        )
        
        ffn_dim = ffn_dim or dim * 4
        
        for i in range(layers):
            self.add_node(PatternNode(
                f"self_attn_{i}", "self_attention", dim, dim,
                {"heads": heads}
            ))
            self.add_node(PatternNode(
                f"ffn_{i}", "ffn", dim, dim,
                {"hidden_dim": ffn_dim}
            ))
        
        self.layers = layers
        self.dim = dim
        self.heads = heads


class Web(Pattern):
    """
    Cross-connected pattern: queries attend to features at multiple scales.
    
    Used for: Colorization, segmentation, conditional generation
    Example: DDColor (100 queries × 3 scales × 9 layers)
    """
    
    def __init__(self, queries: int, dim: int, feature_scales: int, 
                 layers: int, output_dim: int):
        super().__init__(
            name="web",
            topology=Topology.WEB,
            self_reference=SelfReference.PARTIAL,
            io_ratio="N:M"
        )
        
        for i in range(layers):
            scale = i % feature_scales
            self.add_node(PatternNode(
                f"cross_attn_{i}", "cross_attention", dim, dim,
                {"scale": scale, "queries": queries}
            ))
            self.add_node(PatternNode(
                f"self_attn_{i}", "self_attention", dim, dim,
                {"queries": queries}
            ))
            self.add_node(PatternNode(
                f"ffn_{i}", "ffn", dim, dim
            ))
        
        self.add_node(PatternNode("output", "linear", dim, output_dim))
        
        self.queries = queries
        self.dim = dim
        self.layers = layers


class Tree(Pattern):
    """
    Divergent pattern: one input → multiple structured outputs.
    
    Used for: Multi-task learning, ensemble outputs
    Example: Universal scene understanding (image → depth, normals, edges, ...)
    """
    
    def __init__(self, in_dim: int, branches: List[tuple]):
        """
        Args:
            in_dim: Input dimension
            branches: List of (name, out_dim) tuples
        """
        super().__init__(
            name="tree",
            topology=Topology.DIVERGENT,
            self_reference=SelfReference.NONE,
            io_ratio="1:N"
        )
        
        for name, out_dim in branches:
            self.add_node(PatternNode(f"branch_{name}", "linear", in_dim, out_dim))
        
        self.in_dim = in_dim
        self.branches = branches


class Braid(Pattern):
    """
    Intertwined pattern: multiple streams that periodically cross.
    
    Used for: Multi-modal fusion (vision + language)
    """
    
    def __init__(self, streams: List[str], dim: int, layers: int, cross_every: int = 2):
        super().__init__(
            name="braid",
            topology=Topology.BRAID,
            self_reference=SelfReference.PARTIAL,
            io_ratio="N:N"
        )
        
        for i in range(layers):
            for stream in streams:
                self.add_node(PatternNode(
                    f"{stream}_self_{i}", "self_attention", dim, dim
                ))
                
                if i % cross_every == 0:
                    other_streams = [s for s in streams if s != stream]
                    for other in other_streams:
                        self.add_node(PatternNode(
                            f"{stream}_cross_{other}_{i}", "cross_attention", dim, dim,
                            {"from_stream": other}
                        ))
                
                self.add_node(PatternNode(
                    f"{stream}_ffn_{i}", "ffn", dim, dim
                ))
        
        self.streams = streams
        self.dim = dim
        self.layers = layers


class Hourglass(Pattern):
    """
    Symmetric compress/expand pattern with skip connections.
    
    Used for: Autoencoders, U-Net, generation
    """
    
    def __init__(self, dims: List[int], bottleneck_dim: int):
        """
        Args:
            dims: Dimensions at each scale [512, 256, 128, ...]
            bottleneck_dim: Dimension at the narrowest point
        """
        super().__init__(
            name="hourglass",
            topology=Topology.HOURGLASS,
            self_reference=SelfReference.NONE,  # Skip connections, not self-attention
            io_ratio="N:N"
        )
        
        # Encoder (compress)
        prev_dim = dims[0]
        for i, dim in enumerate(dims[1:] + [bottleneck_dim]):
            self.add_node(PatternNode(f"encode_{i}", "linear", prev_dim, dim))
            prev_dim = dim
        
        # Decoder (expand) - reverse order
        for i, dim in enumerate(reversed(dims)):
            self.add_node(PatternNode(
                f"decode_{i}", "linear", prev_dim, dim,
                {"skip_from": f"encode_{len(dims) - i - 1}"}
            ))
            prev_dim = dim
        
        self.dims = dims
        self.bottleneck_dim = bottleneck_dim


class Ring(Pattern):
    """
    Closed loop pattern: output feeds back as input.
    
    Used for: RNNs, state-space models, continuous control
    """
    
    def __init__(self, state_dim: int, input_dim: int, output_dim: int):
        super().__init__(
            name="ring",
            topology=Topology.RING,
            self_reference=SelfReference.RECURRENT,
            io_ratio="1:1"
        )
        
        self.add_node(PatternNode("state_update", "linear", state_dim + input_dim, state_dim))
        self.add_node(PatternNode("output", "linear", state_dim, output_dim))
        
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim


class Constellation(Pattern):
    """
    Graph-structured pattern: nodes with learned connections.
    
    Used for: Knowledge graphs, molecular structure, social networks
    """
    
    def __init__(self, node_dim: int, edge_dim: int, message_passes: int):
        super().__init__(
            name="constellation",
            topology=Topology.CONSTELLATION,
            self_reference=SelfReference.PARTIAL,
            io_ratio="N:M"
        )
        
        for i in range(message_passes):
            self.add_node(PatternNode(
                f"message_{i}", "graph_conv", node_dim, node_dim,
                {"edge_dim": edge_dim}
            ))
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.message_passes = message_passes


class Fractal(Pattern):
    """
    Self-similar pattern at multiple scales.
    
    Used for: Hierarchical structure (document, scene, music)
    """
    
    def __init__(self, dim: int, scales: int, pattern_per_scale: Pattern):
        super().__init__(
            name="fractal",
            topology=Topology.FRACTAL,
            self_reference=SelfReference.PARTIAL,
            io_ratio="N:N"
        )
        
        for scale in range(scales):
            for node in pattern_per_scale.nodes:
                self.add_node(PatternNode(
                    f"scale_{scale}_{node.name}",
                    node.node_type,
                    node.in_dim,
                    node.out_dim,
                    {**node.params, "scale": scale}
                ))
        
        self.dim = dim
        self.scales = scales


class Mirror(Pattern):
    """
    Symmetric encoder-decoder with reflection plane.
    
    Used for: Translation, style transfer, domain adaptation
    """
    
    def __init__(self, encoder_dims: List[int], decoder_dims: List[int]):
        super().__init__(
            name="mirror",
            topology=Topology.MIRROR,
            self_reference=SelfReference.NONE,
            io_ratio="N:M"
        )
        
        # Encoder side
        prev_dim = encoder_dims[0]
        for i, dim in enumerate(encoder_dims[1:]):
            self.add_node(PatternNode(f"encode_{i}", "linear", prev_dim, dim))
            prev_dim = dim
        
        # Decoder side (mirror)
        for i, dim in enumerate(decoder_dims):
            self.add_node(PatternNode(f"decode_{i}", "linear", prev_dim, dim))
            prev_dim = dim
        
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims


# =============================================================================
# PATTERN COMPOSITION
# =============================================================================

def compose(*patterns: Pattern) -> Pattern:
    """
    Compose multiple patterns into a single pattern.
    
    Example:
        multi_task = compose(
            Funnel(1024, 256),  # Shared encoder
            Tree(256, [("depth", 1), ("normals", 3)])  # Multi-task heads
        )
    """
    composed = Pattern(
        name="_".join(p.name for p in patterns),
        topology=patterns[-1].topology,  # Use last pattern's topology
        self_reference=max((p.self_reference for p in patterns), key=lambda x: x.value),
        io_ratio=f"{patterns[0].io_ratio.split(':')[0]}:{patterns[-1].io_ratio.split(':')[1]}"
    )
    
    for pattern in patterns:
        for node in pattern.nodes:
            composed.add_node(node)
    
    return composed


def test_patterns():
    """Test pattern definitions."""
    print("Testing Patterns...")
    
    # Test each pattern type
    patterns = [
        Funnel(1024, 1),
        Spiral(layers=28, dim=3584, heads=28),
        Web(queries=100, dim=256, feature_scales=3, layers=9, output_dim=2),
        Tree(256, [("depth", 1), ("normals", 3), ("edges", 1)]),
        Braid(["vision", "language"], dim=512, layers=6),
        Hourglass([512, 256, 128], bottleneck_dim=64),
        Ring(state_dim=256, input_dim=64, output_dim=32),
    ]
    
    for p in patterns:
        print(f"\n{p.describe()}")
    
    # Test composition
    composed = compose(
        Funnel(1024, 256),
        Tree(256, [("depth", 1), ("normals", 3)])
    )
    print(f"\nComposed:\n{composed.describe()}")
    
    return patterns


if __name__ == "__main__":
    test_patterns()
