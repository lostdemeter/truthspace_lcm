"""
φ-MLP: Gated MLP block for Qwen2-7B.

Architecture:
  gate = normed @ W_gate.T       → (batch, seq, intermediate_size)
  up   = normed @ W_up.T         → (batch, seq, intermediate_size)
  hidden = SiLU(gate) ⊙ up       → element-wise gating
  output = hidden @ W_down.T     → (batch, seq, hidden_dim)
  result = output + residual

SiLU (Swish): x × sigmoid(x)
  In φ-space: sigmoid(x) = 1 / (1 + φ^(-x/ln(φ)))
  This is a φ-level selector — values above 0 pass through, below are suppressed.

No biases on gate/up/down projections in Qwen2-7B.
"""

import numpy as np
from .phi_types import PhiEncoded
from .phi_matmul import phi_linear
from .phi_components import rms_norm, phi_silu


class PhiMLP:
    """
    Gated MLP block for one transformer layer.

    gate_proj: (intermediate_size, hidden_dim) = (18944, 3584)
    up_proj:   (intermediate_size, hidden_dim) = (18944, 3584)
    down_proj: (hidden_dim, intermediate_size) = (3584, 18944)
    """

    def __init__(self, W_gate: PhiEncoded, W_up: PhiEncoded,
                 W_down: PhiEncoded, norm_weight: np.ndarray):
        self.W_gate = W_gate
        self.W_up = W_up
        self.W_down = W_down
        self.norm_weight = norm_weight

    def __call__(self, hidden: np.ndarray, pure: bool = False) -> np.ndarray:
        """
        Forward pass through MLP.

        Args:
            hidden: (batch, seq_len, hidden_dim)
            pure: use pure φ-integer matmul

        Returns:
            output: hidden + mlp_output, same shape
        """
        # Post-attention RMSNorm
        normed = rms_norm(hidden, self.norm_weight)

        # Gate and up projections (no bias)
        gate = phi_linear(self.W_gate, normed, pure=pure)  # (batch, seq, intermediate)
        up = phi_linear(self.W_up, normed, pure=pure)      # (batch, seq, intermediate)

        # SiLU gating
        mlp_hidden = phi_silu(gate) * up

        # Down projection (no bias)
        mlp_output = phi_linear(self.W_down, mlp_hidden, pure=pure)  # (batch, seq, hidden)

        # Residual connection
        return hidden + mlp_output
