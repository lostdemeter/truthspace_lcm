"""
Basic transformer components in φ-space.

- PhiRMSNorm:   RMS normalization (float, not worth φ-encoding)
- PhiEmbedding: Token → φ-vector lookup
- PhiLMHead:    φ-vector → logits
- phi_softmax:  φ-based softmax (φ^(x/T) / Σ φ^(x/T))
- phi_silu:     SiLU activation
"""

import numpy as np
from .phi_types import PhiEncoded, PHI, LOG_PHI
from .phi_matmul import phi_linear


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    RMS normalization: x_normed = x / rms(x) * weight

    This stays in float — normalization is a magnitude operation,
    not a structural one. The structure (signs, relative levels)
    passes through unchanged.
    """
    variance = (x ** 2).mean(axis=-1, keepdims=True)
    x_normed = x / np.sqrt(variance + eps)
    return x_normed * weight


def phi_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax as φ-level selection.

    Standard:  softmax(x) = e^x / Σ e^x
    φ-form:    softmax(x) = φ^(x/T) / Σ φ^(x/T)   where T = ln(φ)

    Since e^x = φ^(x/ln(φ)), this is EXACT — not an approximation.
    The temperature T = ln(φ) ≈ 0.4812 is the natural φ-temperature.

    For numerical stability we subtract the max first (same as standard).
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max

    # e^x = φ^(x / ln(φ))  — exact equivalence
    phi_powers = PHI ** (x_shifted / LOG_PHI)
    return phi_powers / np.sum(phi_powers, axis=axis, keepdims=True)


def phi_silu(x: np.ndarray) -> np.ndarray:
    """
    SiLU (Swish) activation: x * sigmoid(x)

    sigmoid(x) = 1 / (1 + e^(-x)) = 1 / (1 + φ^(-x/ln(φ)))

    The φ-form makes the connection to level selection explicit:
    sigmoid selects between 0 and 1 based on the φ-level of x.
    """
    return x * (1.0 / (1.0 + np.exp(-x)))


class PhiEmbedding:
    """Token embedding lookup table in φ-encoded format."""

    def __init__(self, weight: PhiEncoded):
        # Decode once at init — embeddings are read-only lookup
        self.table = weight.decode()  # (vocab_size, hidden_dim)
        self.vocab_size, self.hidden_dim = self.table.shape

    def __call__(self, token_ids) -> np.ndarray:
        """
        Look up embeddings for token IDs.

        Args:
            token_ids: list or array of token indices

        Returns:
            embeddings: (len(token_ids), hidden_dim) float32
        """
        return self.table[token_ids]


class PhiLMHead:
    """Language model head: hidden → logits via φ-matmul."""

    def __init__(self, weight: PhiEncoded):
        self.weight = weight  # (vocab_size, hidden_dim)

    def __call__(self, hidden: np.ndarray, pure: bool = False) -> np.ndarray:
        """
        Project hidden states to vocabulary logits.

        Args:
            hidden: (batch, seq_len, hidden_dim)
            pure: use pure φ-integer matmul

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        return phi_linear(self.weight, hidden, pure=pure)
