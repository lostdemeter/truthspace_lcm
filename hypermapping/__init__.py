"""
HyperMapping - A Bidirectional Hyperdimensional Data Structure

A new kind of data structure for geometric computation.
Maps inputs to outputs through N-dimensional space.

Usage:
    from hypermapping import HyperMapping, TextEncoder, from_pairs
    
    # Create with text encoder
    encoder = TextEncoder(dims=8)
    encoder.learn(["list files", "show files", "delete file"])
    
    space = HyperMapping(dims=8, encoder=encoder)
    space.map("list files", "ls")
    space.map("delete file", "rm")
    
    # Query
    result = space.forward("show files")  # → "ls"
    results = space.backward("ls")        # → ["list files", ...]
    
    # Pipeline
    pipeline = space1 | space2
    result = pipeline("input")

Author: Lesley Gushurst
License: GPLv3
"""

from .hypermapping import (
    # Core classes
    HyperMapping,
    Mapping,
    MatchResult,
    HyperPipeline,
    
    # Encoders
    Encoder,
    HashEncoder,
    TextEncoder,
    
    # Convenience
    from_pairs,
    
    # Constants
    CRITICAL_LINE,
)

from .encoders import (
    NumericEncoder,
    ImageEncoder,
    CategoricalEncoder,
    CompositeEncoder,
    # Advanced encoders from design docs
    QuaternionEncoder,    # Design 044: 4D semantic axes
    SelfSimilarEncoder,   # Design 072: Function approximation
    SequenceEncoder,      # Design 055: Tachyon navigation
    # Serialization
    ENCODER_REGISTRY,
    encoder_from_dict,
)

__version__ = "0.3.0"  # Serializable encoders
__all__ = [
    # Core
    "HyperMapping",
    "Mapping",
    "MatchResult",
    "HyperPipeline",
    
    # Basic Encoders
    "Encoder",
    "HashEncoder",
    "TextEncoder",
    "NumericEncoder",
    "ImageEncoder",
    "CategoricalEncoder",
    "CompositeEncoder",
    
    # Advanced Encoders (from design docs)
    "QuaternionEncoder",    # Design 044: Sentiment, text classification
    "SelfSimilarEncoder",   # Design 072: Function approximation
    "SequenceEncoder",      # Design 055: Sequence prediction
    
    # Convenience
    "from_pairs",
    
    # Constants
    "CRITICAL_LINE",
    
    # Serialization
    "ENCODER_REGISTRY",
    "encoder_from_dict",
]
