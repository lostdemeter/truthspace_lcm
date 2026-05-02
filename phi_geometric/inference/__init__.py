"""
φ-geometric inference module.

Provides φ-integer encoding and inference for transformer models.

Components:
  phi_types      — PhiEncoded (sign × φ^(exp/128) encoding)
  phi_matmul     — Core matmul (hybrid or pure integer)
  phi_components — RMSNorm, Embedding, LMHead, softmax, SiLU
  phi_attention  — Multi-head attention with RoPE
  phi_mlp        — Gated MLP with SiLU
  phi_engine     — PhiQwen2Engine (full forward pass)
"""

from .phi_types import PhiEncoded, PHI, LOG_PHI, PHI_GRID
from .phi_engine import PhiQwen2Engine
from .phi_integer import (
    GATE_CONTRACT, GATE_PRESERVE_N, GATE_PRESERVE_P, GATE_EXPAND,
    phi_silu_4state,
)

__all__ = [
    'PhiEncoded', 'PHI', 'LOG_PHI', 'PHI_GRID', 'PhiQwen2Engine',
    'GATE_CONTRACT', 'GATE_PRESERVE_N', 'GATE_PRESERVE_P', 'GATE_EXPAND',
    'phi_silu_4state',
]
