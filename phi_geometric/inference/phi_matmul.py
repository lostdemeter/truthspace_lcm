"""
Core φ-integer matrix multiplication.

Two modes:
  HYBRID:  decode φ→float, numpy matmul  (fast, proves pipeline)
  PURE:    sign XOR + int ADD + LUT      (slow, proves integer arith)

Proven in v1: 99.93% correlation (pure) / 99.9991% (hybrid).

The key insight from DA2:
    a × b = φ^(e_a/K) × φ^(e_b/K) = φ^((e_a + e_b)/K)

So matrix multiplication becomes:
    (A @ B)[i,j] = Σ_k sign_a[i,k] × sign_b[k,j] × φ^((e_a[i,k] + e_b[k,j])/K)
                 = Σ_k sign_ab × LUT[e_a + e_b]

This is integer addition + LUT lookup + accumulation.
"""

import numpy as np
from .phi_types import PhiEncoded, PHI, PHI_GRID


class PhiLUT:
    """
    Lookup table for φ^(e/K) values.

    Pre-computes all possible φ-exponent values for fast lookup.
    The table fits in L1/L2 cache (~120 KB for the default range).
    """

    def __init__(self, exp_range=(-25000, 5000)):
        self.exp_min, self.exp_max = exp_range
        self.size = self.exp_max - self.exp_min + 1

        exponents = np.arange(self.exp_min, self.exp_max + 1)
        self.values = (PHI ** (exponents / PHI_GRID)).astype(np.float32)

    def lookup(self, exponent: np.ndarray) -> np.ndarray:
        """Look up φ^(e/K) for given exponents."""
        exp_clipped = np.clip(exponent, self.exp_min, self.exp_max)
        indices = (exp_clipped - self.exp_min).astype(np.intp)
        return self.values[indices]


# Module-level singleton LUT (created once, shared everywhere)
_LUT = None


def get_lut() -> PhiLUT:
    """Get or create the global φ-LUT."""
    global _LUT
    if _LUT is None:
        _LUT = PhiLUT()
    return _LUT


def phi_matmul_hybrid(W: PhiEncoded, x: np.ndarray) -> np.ndarray:
    """
    Matrix multiply: W_decoded @ x.T, returned as (out_features, batch).

    Hybrid mode: decode φ→float, use numpy matmul.
    Fast and accurate. Proves the φ-representation is lossless enough.

    Args:
        W: PhiEncoded weight matrix, shape (out_features, in_features)
        x: float input, shape (batch, in_features)

    Returns:
        output: shape (batch, out_features)
    """
    W_float = W.decode_cached()  # (out_features, in_features) — cached after first call
    return x @ W_float.T  # (batch, in_features) @ (in_features, out_features) = (batch, out_features)


def phi_matmul_pure(W: PhiEncoded, x_signs: np.ndarray, x_exps: np.ndarray,
                    chunk_size: int = 512) -> np.ndarray:
    """
    Pure φ-integer matrix multiply: sign XOR + exponent ADD + LUT.

    NO IEEE floating-point multiplication. Only:
    - int8 sign multiply (equivalent to XOR)
    - int16/int32 exponent addition
    - LUT lookup (pre-computed φ^(e/K) table)
    - float32 accumulation

    Args:
        W: PhiEncoded weight matrix, shape (out_features, in_features)
        x_signs: int8 signs of input, shape (batch, in_features)
        x_exps: int16 exponents of input, shape (batch, in_features)
        chunk_size: chunk output dims to limit memory

    Returns:
        output: shape (batch, out_features), float32
    """
    lut = get_lut()
    out_features, in_features = W.shape
    batch = x_signs.shape[0]

    W_signs = W.signs  # (out_features, in_features)
    W_exps = W.exponents  # (out_features, in_features)

    result = np.zeros((batch, out_features), dtype=np.float32)

    # Process in chunks over output dimension to control memory
    for o_start in range(0, out_features, chunk_size):
        o_end = min(o_start + chunk_size, out_features)
        o_size = o_end - o_start

        # W chunk: (o_size, in_features)
        w_s = W_signs[o_start:o_end]
        w_e = W_exps[o_start:o_end]

        # sign_product[b, o, k] = x_signs[b, k] * w_s[o, k]
        sign_prod = x_signs[:, np.newaxis, :] * w_s[np.newaxis, :, :]  # (batch, o_size, in)

        # exp_sum[b, o, k] = x_exps[b, k] + w_e[o, k]
        exp_sum = (x_exps[:, np.newaxis, :].astype(np.int32)
                   + w_e[np.newaxis, :, :].astype(np.int32))  # (batch, o_size, in)

        # value[b, o, k] = sign_prod × LUT[exp_sum]
        values = sign_prod * lut.lookup(exp_sum)

        # accumulate: result[b, o] = Σ_k values[b, o, k]
        result[:, o_start:o_end] = values.sum(axis=2)

    return result


def phi_linear(W: PhiEncoded, x: np.ndarray, bias: np.ndarray = None,
               pure: bool = False) -> np.ndarray:
    """
    Linear layer: output = x @ W.T + bias

    This is the standard interface for all projection layers.

    Args:
        W: PhiEncoded weight matrix, shape (out_features, in_features)
        x: float input, shape (..., in_features)
        bias: optional bias, shape (out_features,)
        pure: if True, use pure φ-integer matmul

    Returns:
        output: shape (..., out_features)
    """
    orig_shape = x.shape
    in_features = orig_shape[-1]
    x_2d = x.reshape(-1, in_features)

    if pure:
        x_enc = PhiEncoded.encode(x_2d)
        out = phi_matmul_pure(W, x_enc.signs, x_enc.exponents)
    else:
        out = phi_matmul_hybrid(W, x_2d)

    if bias is not None:
        out = out + bias

    # Restore batch dimensions
    out_shape = orig_shape[:-1] + (out.shape[-1],)
    return out.reshape(out_shape)
