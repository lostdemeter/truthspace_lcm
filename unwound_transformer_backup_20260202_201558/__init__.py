"""
Unwound Transformer: Explicit Geometric Computation of Qwen2-7B
================================================================

This module provides a clean, verified implementation of Qwen2-7B
computed entirely through explicit matrix operations. No black boxes.

Key insight: The transformer is fully deterministic. Given weights + tokens,
the output is completely determined by matrix multiplications, RoPE rotations,
softmax, and SiLU activation.

Usage:
    from unwound_transformer import UnwoundQwen2
    
    model = UnwoundQwen2()
    token = model.predict_next(token_A, token_B)
    
    # For geometric analysis:
    trace = model.forward_with_trace(token_A, token_B)
"""

from .model import UnwoundQwen2
from .ops import rms_norm, apply_rope, silu, softmax

__all__ = ['UnwoundQwen2', 'rms_norm', 'apply_rope', 'silu', 'softmax']
