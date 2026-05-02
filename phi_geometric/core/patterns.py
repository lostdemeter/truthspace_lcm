"""
Pattern Taxonomy: Define navigation topologies on the φ-lattice.

Patterns describe HOW information flows through a model:
    - Funnel: Convergent (many → one) - DA2
    - Spiral: Self-referential (deep with attention) - Qwen2-7B
    - Web: Cross-connected (queries attend to features) - DDColor
    - Tree: Divergent (one → many) - Multi-task
    - Braid: Intertwined streams - Multi-modal
    - Hourglass: Compress then expand - Autoencoders
    - Ring: Closed loop (recurrent) - Memory/control
    - Constellation: Graph-structured - Relational
    - Fractal: Self-similar at scales - Hierarchical
    - Mirror: Symmetric translation - Translation

Each pattern is a different shape on the same φ-lattice.
The pattern determines what kind of problem the model can solve.

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class Topology(Enum):
    """The fundamental shape of information flow."""
    CONVERGENT = "convergent"       # Many → One (Funnel)
    DIVERGENT = "divergent"         # One → Many (Tree)
    SPIRAL = "spiral"               # Self-referential loop (Transformer)
    WEB = "web"                     # Cross-connected mesh
    BRAID = "braid"                 # Intertwined parallel streams
    HOURGLASS = "hourglass"         # Compress → Expand
    RING = "ring"                   # Closed recurrent loop
    CONSTELLATION = "constellation" # Graph structure
    FRACTAL = "fractal"             # Self-similar hierarchy
    MIRROR = "mirror"               # Symmetric translation
    CASCADE = "cascade"             # Ordered phases, conditional engagement


class SelfReference(Enum):
    """How much the pattern references itself."""
    NONE = "none"           # Pure feedforward
    PARTIAL = "partial"     # Some self-attention
    FULL = "full"           # Every layer has self-attention
    RECURRENT = "recurrent" # Output feeds back as input


@dataclass
class PatternNode:
    """
    A node in the pattern graph.
    
    Attributes:
        name: Unique identifier for this node
        node_type: Type of operation (linear, attention, ffn, etc.)
        in_dim: Input dimension
        out_dim: Output dimension
        params: Additional parameters for this node type
    """
    name: str
    node_type: str
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
        metadata: Additional pattern-specific data
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
    
    def get_weight_shapes(self) -> Dict[str, tuple]:
        """Get the shapes of all weight tensors in this pattern."""
        shapes = {}
        for node in self.nodes:
            shapes[f"{node.name}.weight"] = (node.out_dim, node.in_dim)
            if node.node_type in ['linear', 'ffn']:
                shapes[f"{node.name}.bias"] = (node.out_dim,)
        return shapes


# =============================================================================
# OBSERVED PATTERNS (from reverse-engineering)
# =============================================================================

class Funnel(Pattern):
    """
    Convergent pattern: many inputs → one output.
    
    Observed in: DA2 (Depth Anything V2)
    
    Characteristics:
        - Simple, focused, one output per location
        - No self-reference
        - Extremely compressible (32 weights for DA2)
    
    Use cases:
        - Depth estimation
        - Classification
        - Regression
    
    Example:
        funnel = Funnel(in_dim=1024, out_dim=1)
        # Creates: input(1024) → hidden(256) → output(1)
    """
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__(
            name="funnel",
            topology=Topology.CONVERGENT,
            self_reference=SelfReference.NONE,
            io_ratio="N:1"
        )
        
        hidden_dim = hidden_dim or max(32, min(in_dim // 4, 256))
        
        self.add_node(PatternNode("compress", "linear", in_dim, hidden_dim))
        self.add_node(PatternNode("output", "linear", hidden_dim, out_dim))
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim


class Spiral(Pattern):
    """
    Self-referential pattern: deep with attention at every layer.
    
    Observed in: Qwen2-7B (Language Model)
    
    Characteristics:
        - Many identical segments (layers)
        - Self-attention at every layer
        - MESH principle applies (pre-compute Q.T @ K)
    
    Use cases:
        - Language modeling
        - Code generation
        - Mathematical reasoning
    
    Example:
        spiral = Spiral(layers=28, dim=3584, heads=28)
        # Creates: 28 layers of (self_attn + ffn)
    """
    
    def __init__(self, layers: int, dim: int, heads: int, ffn_dim: Optional[int] = None):
        super().__init__(
            name="spiral",
            topology=Topology.SPIRAL,
            self_reference=SelfReference.FULL,
            io_ratio="1:1"
        )
        
        ffn_dim = ffn_dim or dim * 4
        head_dim = dim // heads
        
        for i in range(layers):
            # Self-attention
            self.add_node(PatternNode(
                f"self_attn_{i}", "self_attention", dim, dim,
                {"heads": heads, "head_dim": head_dim}
            ))
            # FFN
            self.add_node(PatternNode(
                f"ffn_{i}_up", "linear", dim, ffn_dim
            ))
            self.add_node(PatternNode(
                f"ffn_{i}_down", "linear", ffn_dim, dim
            ))
        
        self.layers = layers
        self.dim = dim
        self.heads = heads
        self.ffn_dim = ffn_dim


class Web(Pattern):
    """
    Cross-connected pattern: queries attend to features at multiple scales.
    
    Observed in: DDColor (Colorization)
    
    Characteristics:
        - Learnable queries that attend to features
        - Cross-attention + self-attention
        - Multi-scale feature processing
    
    Use cases:
        - Colorization
        - Segmentation
        - Conditional generation
    
    Example:
        web = Web(queries=100, dim=256, feature_scales=3, layers=9, output_dim=2)
        # Creates: 100 queries attending to 3 feature scales over 9 layers
    """
    
    def __init__(
        self, 
        queries: int, 
        dim: int, 
        feature_scales: int, 
        layers: int, 
        output_dim: int
    ):
        super().__init__(
            name="web",
            topology=Topology.WEB,
            self_reference=SelfReference.PARTIAL,
            io_ratio="N:M"
        )
        
        for i in range(layers):
            scale = i % feature_scales
            
            # Cross-attention (queries attend to features)
            self.add_node(PatternNode(
                f"cross_attn_{i}", "cross_attention", dim, dim,
                {"scale": scale, "queries": queries}
            ))
            
            # Self-attention (queries attend to each other)
            self.add_node(PatternNode(
                f"self_attn_{i}", "self_attention", dim, dim,
                {"queries": queries}
            ))
            
            # FFN
            self.add_node(PatternNode(
                f"ffn_{i}", "ffn", dim, dim
            ))
        
        # Output projection
        self.add_node(PatternNode("output", "linear", dim, output_dim))
        
        self.queries = queries
        self.dim = dim
        self.feature_scales = feature_scales
        self.layers = layers
        self.output_dim = output_dim


# =============================================================================
# HYPOTHESIZED PATTERNS
# =============================================================================

class Tree(Pattern):
    """
    Divergent pattern: one input → multiple structured outputs.
    
    Use cases:
        - Multi-task learning
        - Ensemble outputs
        - Universal scene understanding
    
    Example:
        tree = Tree(in_dim=256, branches=[
            ("depth", 1),
            ("normals", 3),
            ("edges", 1),
        ])
    """
    
    def __init__(self, in_dim: int, branches: List[tuple]):
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
    
    Use cases:
        - Multi-modal fusion (vision + language)
        - Audio-visual understanding
        - Sensor fusion
    
    Example:
        braid = Braid(
            streams=["vision", "language"],
            dim=512,
            layers=6,
            cross_every=2
        )
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
                # Self-attention within stream
                self.add_node(PatternNode(
                    f"{stream}_self_{i}", "self_attention", dim, dim
                ))
                
                # Cross-attention between streams (every cross_every layers)
                if i % cross_every == 0:
                    for other in streams:
                        if other != stream:
                            self.add_node(PatternNode(
                                f"{stream}_cross_{other}_{i}", "cross_attention", dim, dim,
                                {"from_stream": other}
                            ))
                
                # FFN
                self.add_node(PatternNode(
                    f"{stream}_ffn_{i}", "ffn", dim, dim
                ))
        
        self.streams = streams
        self.dim = dim
        self.layers = layers
        self.cross_every = cross_every


class Hourglass(Pattern):
    """
    Symmetric compress/expand pattern with skip connections.
    
    Use cases:
        - Autoencoders (VAE)
        - Image segmentation (U-Net)
        - Generation / reconstruction
    
    Example:
        hourglass = Hourglass(
            dims=[512, 256, 128],
            bottleneck_dim=64
        )
    """
    
    def __init__(self, dims: List[int], bottleneck_dim: int):
        super().__init__(
            name="hourglass",
            topology=Topology.HOURGLASS,
            self_reference=SelfReference.NONE,
            io_ratio="N:N"
        )
        
        # Encoder (compress)
        prev_dim = dims[0]
        for i, dim in enumerate(dims[1:] + [bottleneck_dim]):
            self.add_node(PatternNode(f"encode_{i}", "linear", prev_dim, dim))
            prev_dim = dim
        
        # Decoder (expand) - reverse order with skip connections
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
    
    Use cases:
        - RNNs, state-space models
        - Continuous control (robotics)
        - Dialogue systems with memory
    
    Example:
        ring = Ring(state_dim=256, input_dim=64, output_dim=32)
    """
    
    def __init__(self, state_dim: int, input_dim: int, output_dim: int):
        super().__init__(
            name="ring",
            topology=Topology.RING,
            self_reference=SelfReference.RECURRENT,
            io_ratio="1:1"
        )
        
        # State update: [state, input] → new_state
        self.add_node(PatternNode(
            "state_update", "linear", state_dim + input_dim, state_dim
        ))
        
        # Output projection: state → output
        self.add_node(PatternNode(
            "output", "linear", state_dim, output_dim
        ))
        
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim


class Constellation(Pattern):
    """
    Graph-structured pattern: nodes with learned connections.
    
    Use cases:
        - Knowledge graphs
        - Molecular structure
        - Social networks
    
    Example:
        constellation = Constellation(
            node_dim=128,
            edge_dim=32,
            message_passes=4
        )
    """
    
    def __init__(self, node_dim: int, edge_dim: int, message_passes: int):
        super().__init__(
            name="constellation",
            topology=Topology.CONSTELLATION,
            self_reference=SelfReference.PARTIAL,
            io_ratio="N:M"
        )
        
        for i in range(message_passes):
            # Message computation
            self.add_node(PatternNode(
                f"message_{i}", "linear", node_dim + edge_dim, node_dim
            ))
            # Node update
            self.add_node(PatternNode(
                f"update_{i}", "linear", node_dim * 2, node_dim
            ))
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.message_passes = message_passes


class Fractal(Pattern):
    """
    Self-similar pattern at multiple scales.
    
    Use cases:
        - Document understanding (word → sentence → paragraph)
        - Scene graphs (object → group → scene)
        - Music (note → phrase → section)
    
    Example:
        fractal = Fractal(dim=256, scales=3, layers_per_scale=2)
    """
    
    def __init__(self, dim: int, scales: int, layers_per_scale: int = 2):
        super().__init__(
            name="fractal",
            topology=Topology.FRACTAL,
            self_reference=SelfReference.PARTIAL,
            io_ratio="N:N"
        )
        
        for scale in range(scales):
            for layer in range(layers_per_scale):
                # Same pattern at each scale
                self.add_node(PatternNode(
                    f"scale_{scale}_attn_{layer}", "self_attention", dim, dim,
                    {"scale": scale}
                ))
                self.add_node(PatternNode(
                    f"scale_{scale}_ffn_{layer}", "ffn", dim, dim,
                    {"scale": scale}
                ))
            
            # Cross-scale connection (except last scale)
            if scale < scales - 1:
                self.add_node(PatternNode(
                    f"scale_{scale}_pool", "linear", dim, dim,
                    {"pooling": True}
                ))
        
        self.dim = dim
        self.scales = scales
        self.layers_per_scale = layers_per_scale


class Mirror(Pattern):
    """
    Symmetric encoder-decoder with reflection plane.
    
    Use cases:
        - Language translation
        - Style transfer
        - Domain adaptation
    
    Example:
        mirror = Mirror(
            encoder_dims=[512, 256, 128],
            decoder_dims=[128, 256, 512]
        )
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


class Cascade(Pattern):
    """
    Ordered phase pipeline with conditional engagement.
    
    Discovered in: IPA geometric demo (English → IPA transformation)
    
    Characteristics:
        - Multiple ordered phases, executed sequentially
        - Each phase has multiple rules; only one fires per input
        - Rules are selected by context (gear discovery)
        - Outputs can be FROZEN to prevent re-processing by later phases
        - Context is evaluated against the ORIGINAL input, not intermediates
    
    This is the topology of a tumbler lock:
        - Each phase is a disk with rule spokes
        - The input (key) passes through each disk
        - Each disk selects one transformation to apply
        - The composition of all phases produces the output
    
    Use cases:
        - Sequential transformation pipelines
        - Phonological rule systems (IPA, transliteration)
        - Multi-pass compilation / optimization
        - Any system where order of operations matters
    
    Example:
        cascade = Cascade(
            phases=[
                Phase('detect', rules=['magic_e', 'igh', 'silent_e']),
                Phase('collapse', rules=['sh', 'th', 'ng', 'ch', ...]),
                Phase('context', rules=['c_rule', 'g_rule', 'y_rule']),
                Phase('charmap', rules=['a→æ', 'e→ɛ', 'i→ɪ', ...]),
            ],
            preserve_original_context=True
        )
    """
    
    def __init__(
        self,
        phases: List[Dict],
        dim: int = 256,
        preserve_original_context: bool = True
    ):
        super().__init__(
            name="cascade",
            topology=Topology.CASCADE,
            self_reference=SelfReference.NONE,
            io_ratio="1:1",
            metadata={
                'preserve_original_context': preserve_original_context,
                'n_phases': len(phases),
                'phase_names': [p.get('name', f'phase_{i}') for i, p in enumerate(phases)],
            }
        )
        
        for i, phase in enumerate(phases):
            name = phase.get('name', f'phase_{i}')
            n_rules = phase.get('n_rules', len(phase.get('rules', [])))
            
            # Selector: picks which rule fires
            self.add_node(PatternNode(
                f"{name}_selector", "selector", dim, n_rules,
                {
                    'phase_index': i,
                    'rules': phase.get('rules', []),
                    'frozen_outputs': phase.get('frozen_outputs', False),
                }
            ))
            
            # Transform: applies the selected rule
            self.add_node(PatternNode(
                f"{name}_transform", "linear", dim, dim,
                {'phase_index': i, 'conditional': True}
            ))
        
        self.phases = phases
        self.dim = dim
        self.preserve_original_context = preserve_original_context
    
    @property
    def n_phases(self) -> int:
        return len(self.phases)
    
    @property
    def total_rules(self) -> int:
        return sum(
            p.get('n_rules', len(p.get('rules', [])))
            for p in self.phases
        )


# =============================================================================
# PATTERN COMPOSITION
# =============================================================================

def compose(*patterns: Pattern) -> Pattern:
    """
    Compose multiple patterns into a single pattern.
    
    The composed pattern chains the patterns sequentially,
    inheriting the topology of the last pattern.
    
    Example:
        # Shared encoder + multi-task heads
        multi_task = compose(
            Funnel(1024, 256),  # Shared encoder
            Tree(256, [("depth", 1), ("normals", 3)])  # Multi-task heads
        )
    """
    composed = Pattern(
        name="_".join(p.name for p in patterns),
        topology=patterns[-1].topology,
        self_reference=max(
            (p.self_reference for p in patterns), 
            key=lambda x: list(SelfReference).index(x)
        ),
        io_ratio=f"{patterns[0].io_ratio.split(':')[0]}:{patterns[-1].io_ratio.split(':')[1]}"
    )
    
    for pattern in patterns:
        for node in pattern.nodes:
            composed.add_node(node)
    
    return composed


# =============================================================================
# PATTERN SELECTION
# =============================================================================

def select_pattern_for_problem(
    input_types: List[str],
    output_types: List[str],
    temporal: bool = False,
    hierarchical: bool = False,
    cross_modal: bool = False,
    symmetric: bool = False,
    sequential_rules: bool = False
) -> str:
    """
    Select the appropriate pattern based on problem characteristics.
    
    Returns the pattern name (funnel, spiral, web, etc.)
    """
    # Sequential rule application (ordered phases) → Cascade
    if sequential_rules:
        return "cascade"
    
    # Multiple outputs from single input → Tree
    if len(output_types) > 1 and len(input_types) == 1:
        return "tree"
    
    # Cross-modal or multiple input types → Braid or Web
    if cross_modal or len(set(input_types)) > 1:
        if len(input_types) > 1:
            return "braid"
        return "web"
    
    # Temporal/sequential → Spiral
    if temporal:
        return "spiral"
    
    # Symmetric (autoencoder-like) → Hourglass
    if symmetric:
        return "hourglass"
    
    # Hierarchical → Fractal
    if hierarchical:
        return "fractal"
    
    # Default: Funnel (simple prediction)
    return "funnel"


def test_patterns():
    """Test pattern definitions."""
    print("=" * 60)
    print("PATTERN TAXONOMY TEST")
    print("=" * 60)
    
    patterns = [
        Funnel(1024, 1),
        Spiral(layers=4, dim=256, heads=4),
        Web(queries=10, dim=64, feature_scales=2, layers=3, output_dim=2),
        Tree(256, [("a", 1), ("b", 3)]),
        Braid(["v", "l"], dim=128, layers=2),
        Hourglass([256, 128], bottleneck_dim=32),
        Ring(state_dim=64, input_dim=16, output_dim=8),
    ]
    
    for p in patterns:
        print(f"\n{p.describe()}")
        shapes = p.get_weight_shapes()
        total_params = sum(
            s[0] * s[1] if len(s) == 2 else s[0] 
            for s in shapes.values()
        )
        print(f"  Total params: {total_params:,}")
    
    # Test composition
    print("\n--- Composition ---")
    composed = compose(
        Funnel(1024, 256),
        Tree(256, [("depth", 1), ("normals", 3)])
    )
    print(f"\n{composed.describe()}")
    
    print("\n" + "=" * 60)
    print("PATTERN TAXONOMY TEST COMPLETE")
    print("=" * 60)
    
    return patterns


if __name__ == "__main__":
    test_patterns()
