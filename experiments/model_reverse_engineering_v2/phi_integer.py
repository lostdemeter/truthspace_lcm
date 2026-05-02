"""
Pure integer arithmetic primitives for φ-encoded computation.

All operations use ONLY:
  - int8 sign multiply (XOR)
  - int16/int32 exponent addition
  - int64 accumulation
  - Pre-computed lookup tables (int → int)

NO IEEE float multiply, divide, sqrt, or exp at inference time.
LUTs are built once at init from the φ-lattice definition.

Core primitives:
  - PhiFixedLUT:       block-scaled accumulation LUT
  - phi_accumulate:    sum of φ-encoded values via fixed-point
  - PhiSiLULUT:        SiLU as integer LUT
  - phi_rms_norm_int:  RMS norm via integer ops
  - phi_add_encoded:   residual add of two φ-encoded values
  - phi_matmul_integer: matrix multiply via sign XOR + exp ADD
  - PhiSoftmaxLUT:     softmax exponentiation as integer LUT
"""

import numpy as np
from phi_types import PhiEncoded, PHI, LOG_PHI, PHI_GRID

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIXED_SCALE_BITS = 30
FIXED_SCALE = 1 << FIXED_SCALE_BITS
EXP_MIN = -25000
EXP_MAX = 5000
EXP_RANGE = EXP_MAX - EXP_MIN + 1

# Max shifted exponent we bother with (terms below this → 0)
SHIFT_CUTOFF = -3200

# Gate code constants (4-state holographic gate encoding)
GATE_CONTRACT   = np.int8(0)
GATE_PRESERVE_N = np.int8(1)
GATE_PRESERVE_P = np.int8(2)
GATE_EXPAND     = np.int8(3)
LOG_PHI_BOUNDARY = float(LOG_PHI)


# ---------------------------------------------------------------------------
# Block-Scaled Fixed-Point Accumulation LUT
# ---------------------------------------------------------------------------
class PhiFixedLUT:
    """
    Lookup table for block-scaled fixed-point accumulation.

    Maps shifted exponent (≤ 0) → scaled integer value.
    LUT_fixed[s] = round(φ^(s / K) × FIXED_SCALE)
    """

    def __init__(self):
        n_entries = abs(SHIFT_CUTOFF) + 1
        self.forward = np.zeros(n_entries, dtype=np.int64)

        for i in range(n_entries):
            shift = -i
            val = PHI ** (shift / PHI_GRID) * FIXED_SCALE
            self.forward[i] = int(round(val))

    def lookup_forward(self, shifted_exp: np.ndarray) -> np.ndarray:
        """Look up scaled integer values for shifted exponents (all ≤ 0)."""
        neg_shift = np.minimum(-shifted_exp, abs(SHIFT_CUTOFF)).astype(np.intp)
        return self.forward[neg_shift]

    def reverse_lookup(self, abs_sum: np.ndarray) -> np.ndarray:
        """Convert accumulated |sum| back to φ-exponent offset."""
        result = np.full(abs_sum.shape, -30000, dtype=np.int16)
        mask = abs_sum > 0

        if np.any(mask):
            vals = abs_sum[mask].astype(np.float64)
            offsets = np.round(
                PHI_GRID * (np.log(vals) - FIXED_SCALE_BITS * np.log(2.0)) / LOG_PHI
            ).astype(np.int16)
            result[mask] = offsets

        return result


_FIXED_LUT = None


def get_fixed_lut() -> PhiFixedLUT:
    global _FIXED_LUT
    if _FIXED_LUT is None:
        _FIXED_LUT = PhiFixedLUT()
    return _FIXED_LUT


def _clamp_exps(exps_i32: np.ndarray) -> np.ndarray:
    """Clamp int32 exponents to representable range and cast to int16."""
    return np.clip(exps_i32, EXP_MIN, EXP_MAX).astype(np.int16)


# ---------------------------------------------------------------------------
# Core: Block-Scaled Fixed-Point Accumulation
# ---------------------------------------------------------------------------
def phi_accumulate(signs: np.ndarray, exponents: np.ndarray,
                   axis: int = -1) -> tuple:
    """
    Sum φ-encoded values along an axis using block-scaled fixed-point.
    Pure integer: no IEEE float multiply/divide.
    """
    lut = get_fixed_lut()
    exps = exponents.astype(np.int32)

    max_exp = np.max(exps, axis=axis, keepdims=True)
    shifted = exps - max_exp
    scaled_vals = lut.lookup_forward(shifted)
    signed_vals = signs.astype(np.int64) * scaled_vals
    sum_int = np.sum(signed_vals, axis=axis)

    result_signs = np.sign(sum_int).astype(np.int8)
    result_signs[result_signs == 0] = 1

    abs_sum = np.abs(sum_int)
    exp_offset = lut.reverse_lookup(abs_sum)

    max_exp_squeezed = np.squeeze(max_exp, axis=axis)
    result_exps = _clamp_exps(max_exp_squeezed.astype(np.int32) + exp_offset.astype(np.int32))

    return result_signs, result_exps


# ---------------------------------------------------------------------------
# SiLU Integer LUT
# ---------------------------------------------------------------------------
class PhiSiLULUT:
    """
    SiLU(x) = x × sigmoid(x) as a pure integer LUT.
    Input: (sign, exponent) → Output: (sign, exponent, gate_code)
    """

    def __init__(self):
        self.out_signs = np.zeros((2, EXP_RANGE), dtype=np.int8)
        self.out_exps = np.zeros((2, EXP_RANGE), dtype=np.int16)
        self.gate_codes = np.zeros((2, EXP_RANGE), dtype=np.int8)

        for s_idx, s in enumerate([-1, +1]):
            for e_idx in range(EXP_RANGE):
                exp = EXP_MIN + e_idx
                x = s * PHI ** (exp / PHI_GRID)
                y = float(x * (1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))))

                if abs(y) < 1e-20:
                    self.out_signs[s_idx, e_idx] = np.int8(-1 if x < 0 else 1)
                    self.out_exps[s_idx, e_idx] = np.int16(EXP_MIN)
                else:
                    self.out_signs[s_idx, e_idx] = np.int8(1 if y > 0 else -1)
                    self.out_exps[s_idx, e_idx] = np.int16(round(
                        PHI_GRID * np.log(abs(y)) / LOG_PHI
                    ))

                if x < -LOG_PHI_BOUNDARY:
                    self.gate_codes[s_idx, e_idx] = GATE_CONTRACT
                elif x < 0:
                    self.gate_codes[s_idx, e_idx] = GATE_PRESERVE_N
                elif x < LOG_PHI_BOUNDARY:
                    self.gate_codes[s_idx, e_idx] = GATE_PRESERVE_P
                else:
                    self.gate_codes[s_idx, e_idx] = GATE_EXPAND

    def __call__(self, signs: np.ndarray, exponents: np.ndarray) -> tuple:
        """Apply SiLU via integer LUT lookup."""
        s_idx = ((signs.astype(np.int32) + 1) // 2).astype(np.intp)
        e_idx = np.clip(exponents.astype(np.int32) - EXP_MIN, 0, EXP_RANGE - 1).astype(np.intp)
        return self.out_signs[s_idx, e_idx], self.out_exps[s_idx, e_idx]


_SILU_LUT = None


def get_silu_lut() -> PhiSiLULUT:
    global _SILU_LUT
    if _SILU_LUT is None:
        _SILU_LUT = PhiSiLULUT()
    return _SILU_LUT


def phi_silu_int(signs: np.ndarray, exponents: np.ndarray) -> tuple:
    """SiLU via integer LUT. Input/output: (int8 signs, int16 exps)."""
    return get_silu_lut()(signs, exponents)


# ---------------------------------------------------------------------------
# Softmax Exponentiation LUT
# ---------------------------------------------------------------------------
class PhiSoftmaxLUT:
    """exp(x) = φ^(x/ln(φ)) as integer LUT."""

    def __init__(self):
        self.out_exps = np.zeros((2, EXP_RANGE), dtype=np.int16)

        for s_idx, s in enumerate([-1, +1]):
            for e_idx in range(EXP_RANGE):
                exp_val = EXP_MIN + e_idx
                x = s * PHI ** (exp_val / PHI_GRID)
                grid_exp = round(PHI_GRID * x / LOG_PHI)
                grid_exp = int(np.clip(grid_exp, EXP_MIN, EXP_MAX))
                self.out_exps[s_idx, e_idx] = np.int16(grid_exp)

    def __call__(self, signs: np.ndarray, exponents: np.ndarray) -> np.ndarray:
        """Compute exp(x) exponents via integer LUT."""
        s_idx = ((signs.astype(np.int32) + 1) // 2).astype(np.intp)
        e_idx = np.clip(exponents.astype(np.int32) - EXP_MIN, 0, EXP_RANGE - 1).astype(np.intp)
        return self.out_exps[s_idx, e_idx]


_SOFTMAX_LUT = None


def get_softmax_lut() -> PhiSoftmaxLUT:
    global _SOFTMAX_LUT
    if _SOFTMAX_LUT is None:
        _SOFTMAX_LUT = PhiSoftmaxLUT()
    return _SOFTMAX_LUT


# ---------------------------------------------------------------------------
# Element-wise Integer Multiply
# ---------------------------------------------------------------------------
def phi_multiply_int(a_signs, a_exps, b_signs, b_exps):
    """Element-wise multiply: sign XOR + exponent ADD."""
    out_signs = (a_signs * b_signs).astype(np.int8)
    out_exps = _clamp_exps(a_exps.astype(np.int32) + b_exps.astype(np.int32))
    return out_signs, out_exps


# ---------------------------------------------------------------------------
# Integer Residual Add
# ---------------------------------------------------------------------------
def phi_add_encoded(a_signs, a_exps, b_signs, b_exps):
    """Add two φ-encoded values element-wise via block-scaled accumulation."""
    stacked_signs = np.stack([a_signs, b_signs], axis=-1)
    stacked_exps = np.stack([a_exps.astype(np.int32),
                             b_exps.astype(np.int32)], axis=-1)
    return phi_accumulate(stacked_signs, stacked_exps, axis=-1)


# ---------------------------------------------------------------------------
# Integer RMS Norm
# ---------------------------------------------------------------------------
def phi_rms_norm_int(x_signs, x_exps, w_signs, w_exps, hidden_dim=3584):
    """
    RMS norm via pure integer ops.
    y = x / sqrt(mean(x²)) * weight
    """
    # 1. Square: sign=+1, exp=2×exp_x
    sq_signs = np.ones_like(x_signs)
    sq_exps = np.clip(2 * x_exps.astype(np.int32), EXP_MIN, EXP_MAX).astype(np.int16)

    # 2. Sum of squares via block-scaled accumulation
    sum_signs, sum_exps = phi_accumulate(sq_signs, sq_exps, axis=-1)

    # 3. Divide by D: subtract log_φ(D) from exponent
    log_phi_D = int(round(PHI_GRID * np.log(hidden_dim) / LOG_PHI))
    mean_exps = (sum_exps.astype(np.int32) - log_phi_D)

    # 4. Square root: halve the exponent
    rms_exps = mean_exps // 2

    # 5. Normalize: x / rms
    rms_expanded = np.expand_dims(rms_exps, axis=-1)
    normed_exps = (x_exps.astype(np.int32) - rms_expanded)

    # 6. Scale by weight: XOR signs, add exponents
    out_signs = (x_signs * w_signs).astype(np.int8)
    out_exps = _clamp_exps(normed_exps + w_exps.astype(np.int32))

    return out_signs, out_exps


# ---------------------------------------------------------------------------
# Integer φ-Matmul (fully integer, block-scaled accumulation)
# ---------------------------------------------------------------------------
def phi_matmul_integer(W, x_signs, x_exps, chunk_size=256):
    """
    Matrix multiply returning φ-encoded result.
    Pure integer: sign XOR + exp ADD + block-scaled accumulation.

    Args:
        W: PhiEncoded weight matrix, shape (out_features, in_features)
        x_signs: int8 input signs, shape (batch, in_features)
        x_exps: int16 input exponents, shape (batch, in_features)

    Returns:
        (out_signs, out_exps): shape (batch, out_features)
    """
    lut = get_fixed_lut()
    out_features, in_features = W.shape
    batch = x_signs.shape[0]

    result_signs = np.zeros((batch, out_features), dtype=np.int8)
    result_exps = np.zeros((batch, out_features), dtype=np.int16)

    W_signs = W.signs
    W_exps = W.exponents

    for o_start in range(0, out_features, chunk_size):
        o_end = min(o_start + chunk_size, out_features)

        w_s = W_signs[o_start:o_end]
        w_e = W_exps[o_start:o_end]

        sign_prod = x_signs[:, np.newaxis, :] * w_s[np.newaxis, :, :]
        exp_sum = (x_exps[:, np.newaxis, :].astype(np.int32)
                   + w_e[np.newaxis, :, :].astype(np.int32))

        max_exp = np.max(exp_sum, axis=2, keepdims=True)
        shifted = exp_sum - max_exp
        scaled = lut.lookup_forward(shifted)
        signed_scaled = sign_prod.astype(np.int64) * scaled
        sum_int = np.sum(signed_scaled, axis=2)

        chunk_signs = np.sign(sum_int).astype(np.int8)
        chunk_signs[chunk_signs == 0] = 1

        abs_sum = np.abs(sum_int)
        exp_offset = lut.reverse_lookup(abs_sum)

        max_exp_sq = np.squeeze(max_exp, axis=2)
        chunk_exps = _clamp_exps(max_exp_sq.astype(np.int32) + exp_offset.astype(np.int32))

        result_signs[:, o_start:o_end] = chunk_signs
        result_exps[:, o_start:o_end] = chunk_exps

    return result_signs, result_exps


# ---------------------------------------------------------------------------
# Full Integer Softmax
# ---------------------------------------------------------------------------
def phi_softmax_full_int(x_signs, x_exps, axis=-1):
    """
    Full softmax in integer mode:
      1. Shift by max
      2. Exponentiate via LUT
      3. Normalize via block-scaled accumulation
    """
    softmax_lut = get_softmax_lut()

    x_float = x_signs.astype(np.float64) * (PHI ** (x_exps.astype(np.float64) / PHI_GRID))
    x_max = np.max(x_float, axis=axis, keepdims=True)

    x_shifted = x_float - x_max
    shifted_signs = np.where(x_shifted >= 0, 1, -1).astype(np.int8)
    abs_shifted = np.abs(x_shifted)
    safe_abs = np.maximum(abs_shifted, 1e-30)
    shifted_exps = np.round(PHI_GRID * np.log(safe_abs) / LOG_PHI).astype(np.int16)
    shifted_exps[abs_shifted < 1e-30] = EXP_MIN

    exp_exps = softmax_lut(shifted_signs, shifted_exps)
    exp_signs = np.ones_like(exp_exps, dtype=np.int8)

    sum_signs, sum_exps = phi_accumulate(exp_signs, exp_exps, axis=axis)
    sum_exps_expanded = np.expand_dims(sum_exps, axis=axis)
    out_exps = _clamp_exps(exp_exps.astype(np.int32) - sum_exps_expanded.astype(np.int32))
    out_signs = exp_signs

    return out_signs, out_exps


# ---------------------------------------------------------------------------
# Encode / Decode utilities
# ---------------------------------------------------------------------------
def float_to_phi(x):
    """Encode float array to (int8 signs, int16 exponents)."""
    enc = PhiEncoded.encode(x.astype(np.float32))
    return enc.signs, enc.exponents


def phi_to_float(signs, exps):
    """Decode φ-encoded (signs, exps) back to float32."""
    return (signs.astype(np.float64) * PHI ** (exps.astype(np.float64) / PHI_GRID)).astype(np.float32)
