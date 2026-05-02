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
  - PhiFixedLUT:     block-scaled accumulation LUT
  - phi_accumulate:  sum of φ-encoded values via fixed-point
  - phi_silu_lut:    SiLU as integer LUT
  - phi_rms_norm_int: RMS norm via integer ops
  - phi_add_encoded: residual add of two φ-encoded values
  - phi_softmax_lut: softmax exponentiation as integer LUT
"""

import numpy as np
from .phi_types import PhiEncoded, PHI, LOG_PHI, PHI_GRID

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIXED_SCALE_BITS = 30          # 2^30 = 1,073,741,824 scale factor
FIXED_SCALE = 1 << FIXED_SCALE_BITS
EXP_MIN = -25000
EXP_MAX = 5000
EXP_RANGE = EXP_MAX - EXP_MIN + 1

# Max shifted exponent we bother with (terms below this → 0)
# φ^(-3200/128) = φ^(-25) ≈ 1.2e-5, negligible relative to dominant term
SHIFT_CUTOFF = -3200

# Gate code constants (Doc 253/254: 4-state holographic gate encoding)
# Boundaries at ±log(φ) ≈ ±0.481 on the INPUT to SiLU
GATE_CONTRACT   = np.int8(0)   # x < -log(φ): deep negative, exponential suppression
GATE_PRESERVE_N = np.int8(1)   # -log(φ) ≤ x < 0: negative zero, linear regime
GATE_PRESERVE_P = np.int8(2)   # 0 ≤ x < +log(φ): positive zero, linear regime
GATE_EXPAND     = np.int8(3)   # x ≥ +log(φ): full fire, identity regime
LOG_PHI_BOUNDARY = float(LOG_PHI)  # ≈ 0.481


# ---------------------------------------------------------------------------
# Block-Scaled Fixed-Point Accumulation LUT
# ---------------------------------------------------------------------------
class PhiFixedLUT:
    """
    Lookup table for block-scaled fixed-point accumulation.

    Maps shifted exponent (≤ 0) → scaled integer value.
    LUT_fixed[s] = round(φ^(s / K) × FIXED_SCALE)

    For s=0: value = FIXED_SCALE (the dominant term)
    For s<0: value decreases geometrically
    For s<SHIFT_CUTOFF: value = 0 (negligible)
    """

    def __init__(self):
        # Forward LUT: shifted exponent → scaled integer
        # Only need non-positive shifts (shifted = exp - max_exp ≤ 0)
        n_entries = abs(SHIFT_CUTOFF) + 1  # 0 to SHIFT_CUTOFF
        self.forward = np.zeros(n_entries, dtype=np.int64)

        for i in range(n_entries):
            shift = -i  # 0, -1, -2, ..., SHIFT_CUTOFF
            val = PHI ** (shift / PHI_GRID) * FIXED_SCALE
            self.forward[i] = int(round(val))

        # Reverse lookup uses vectorized np.log (see reverse_lookup method)

    def lookup_forward(self, shifted_exp: np.ndarray) -> np.ndarray:
        """
        Look up scaled integer values for shifted exponents.

        Args:
            shifted_exp: int32 array, all values ≤ 0

        Returns:
            int64 array of scaled integer values
        """
        # Clip to valid range and negate (our table is indexed by -shift)
        neg_shift = np.minimum(-shifted_exp, abs(SHIFT_CUTOFF)).astype(np.intp)
        return self.forward[neg_shift]

    def reverse_lookup(self, abs_sum: np.ndarray) -> np.ndarray:
        """
        Convert accumulated |sum| back to φ-exponent offset (VECTORIZED).

        Uses: round(PHI_GRID * ln(|sum| / SCALE) / LOG_PHI)
        Implemented via float64 log to get the exponent, then round.
        The float64 log is used ONLY for re-encoding — not for any
        structural computation. It replaces the slow Python bit_length loop.

        Args:
            abs_sum: int64 array of absolute accumulated values (> 0)

        Returns:
            int16 array of exponent offsets relative to block max
        """
        result = np.full(abs_sum.shape, -30000, dtype=np.int16)
        mask = abs_sum > 0

        if np.any(mask):
            vals = abs_sum[mask].astype(np.float64)
            # exp_offset = round(PHI_GRID * ln(val / SCALE) / LOG_PHI)
            offsets = np.round(
                PHI_GRID * (np.log(vals) - FIXED_SCALE_BITS * np.log(2.0)) / LOG_PHI
            ).astype(np.int16)
            result[mask] = offsets

        return result


# Module-level singleton
_FIXED_LUT = None


def get_fixed_lut() -> PhiFixedLUT:
    global _FIXED_LUT
    if _FIXED_LUT is None:
        _FIXED_LUT = PhiFixedLUT()
    return _FIXED_LUT


# ---------------------------------------------------------------------------
# Core: Block-Scaled Fixed-Point Accumulation
# ---------------------------------------------------------------------------
def phi_accumulate(signs: np.ndarray, exponents: np.ndarray,
                   axis: int = -1) -> tuple:
    """
    Sum φ-encoded values along an axis using block-scaled fixed-point.

    Pure integer: no IEEE float multiply/divide.

    Args:
        signs: int8 array of signs (-1 or +1)
        exponents: int16/int32 array of exponents
        axis: axis to sum along

    Returns:
        (result_signs, result_exponents): φ-encoded sum
        Both are arrays with the summed axis removed.
    """
    lut = get_fixed_lut()
    exps = exponents.astype(np.int32)

    # 1. Find max exponent along axis (the dominant term)
    max_exp = np.max(exps, axis=axis, keepdims=True)  # int32 max

    # 2. Shift all exponents relative to max
    shifted = exps - max_exp  # int32, all ≤ 0

    # 3. Look up scaled integer values
    scaled_vals = lut.lookup_forward(shifted)  # int64

    # 4. Apply signs
    signed_vals = signs.astype(np.int64) * scaled_vals  # int64

    # 5. Accumulate (pure integer addition)
    sum_int = np.sum(signed_vals, axis=axis)  # int64

    # 6. Extract result sign
    result_signs = np.sign(sum_int).astype(np.int8)
    result_signs[result_signs == 0] = 1  # zero → positive

    # 7. Reverse lookup: |sum| → exponent offset
    abs_sum = np.abs(sum_int)
    exp_offset = lut.reverse_lookup(abs_sum)

    # 8. Result exponent = max_exp + offset (clamp to valid range)
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
    Built once at init. ~270 KB. Fits in L2 cache.

    Gate code classifies each input into one of 4 holographic states
    based on ±log(φ) boundaries (Doc 253/254):
      0 = CONTRACT:   x < -log(φ),  deep negative, exponential suppression
      1 = PRESERVE-:  -log(φ) ≤ x < 0,  negative zero, linear regime
      2 = PRESERVE+:  0 ≤ x < +log(φ),  positive zero, linear regime
      3 = EXPAND:     x ≥ +log(φ),  full fire, identity regime
    """

    def __init__(self):
        # For positive and negative inputs
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

                # 4-state gate code from INPUT boundaries
                if x < -LOG_PHI_BOUNDARY:
                    self.gate_codes[s_idx, e_idx] = GATE_CONTRACT
                elif x < 0:
                    self.gate_codes[s_idx, e_idx] = GATE_PRESERVE_N
                elif x < LOG_PHI_BOUNDARY:
                    self.gate_codes[s_idx, e_idx] = GATE_PRESERVE_P
                else:
                    self.gate_codes[s_idx, e_idx] = GATE_EXPAND

    def __call__(self, signs: np.ndarray, exponents: np.ndarray) -> tuple:
        """
        Apply SiLU via integer LUT lookup.

        Args:
            signs: int8 array (-1 or +1)
            exponents: int16 array

        Returns:
            (out_signs, out_exps): SiLU(x) in φ-encoded form
        """
        s_idx = ((signs.astype(np.int32) + 1) // 2).astype(np.intp)  # -1→0, +1→1
        e_idx = np.clip(exponents.astype(np.int32) - EXP_MIN, 0, EXP_RANGE - 1).astype(np.intp)

        out_s = self.out_signs[s_idx, e_idx]
        out_e = self.out_exps[s_idx, e_idx]
        return out_s, out_e

    def __call_4state__(self, signs: np.ndarray, exponents: np.ndarray) -> tuple:
        """
        Apply SiLU with 4-state gate code output.

        Returns:
            (out_signs, out_exps, gate_codes): SiLU output + holographic gate state
        """
        s_idx = ((signs.astype(np.int32) + 1) // 2).astype(np.intp)
        e_idx = np.clip(exponents.astype(np.int32) - EXP_MIN, 0, EXP_RANGE - 1).astype(np.intp)

        out_s = self.out_signs[s_idx, e_idx]
        out_e = self.out_exps[s_idx, e_idx]
        gate_c = self.gate_codes[s_idx, e_idx]
        return out_s, out_e, gate_c


_SILU_LUT = None


def get_silu_lut() -> PhiSiLULUT:
    global _SILU_LUT
    if _SILU_LUT is None:
        _SILU_LUT = PhiSiLULUT()
    return _SILU_LUT


def phi_silu_int(signs: np.ndarray, exponents: np.ndarray) -> tuple:
    """SiLU via integer LUT. Input/output: (int8 signs, int16 exps)."""
    return get_silu_lut()(signs, exponents)


def phi_silu_4state(signs: np.ndarray, exponents: np.ndarray) -> tuple:
    """
    SiLU with 4-state gate code (Doc 253/254).

    Returns:
        (out_signs, out_exps, gate_codes): SiLU output + holographic gate state
        gate_codes: int8 array with values 0-3:
          0=CONTRACT, 1=PRESERVE-, 2=PRESERVE+, 3=EXPAND
    """
    return get_silu_lut().__call_4state__(signs, exponents)


# ---------------------------------------------------------------------------
# Softmax Exponentiation LUT
# ---------------------------------------------------------------------------
class PhiSoftmaxLUT:
    """
    Softmax exponentiation: exp(x) = φ^(x/ln(φ)) as integer LUT.

    Input: (sign, exponent) of x → Output: exponent of exp(x)
    Note: exp(x) is always positive, so output sign is always +1.

    For softmax normalization, we also need accumulation (phi_accumulate).
    """

    def __init__(self):
        self.out_exps = np.zeros((2, EXP_RANGE), dtype=np.int16)

        for s_idx, s in enumerate([-1, +1]):
            for e_idx in range(EXP_RANGE):
                exp_val = EXP_MIN + e_idx
                x = s * PHI ** (exp_val / PHI_GRID)
                # exp(x) = φ^(x/ln(φ))
                # exponent = round(K × x / ln(φ)) = round(K × s × φ^(exp_val/K) / ln(φ))
                y_exp = x / LOG_PHI  # This is the φ-exponent of exp(x)
                # Clamp to representable range
                phi_exp = int(round(np.clip(y_exp * PHI_GRID, EXP_MIN, EXP_MAX)))
                # Wait — we need the exponent in the φ-grid.
                # exp(x) = φ^(x/ln(φ)). The φ-exponent is x/ln(φ).
                # In terms of grid: grid_exp = round(PHI_GRID * x / LOG_PHI)
                # But x = s * PHI^(exp_val/PHI_GRID), so:
                # grid_exp = round(PHI_GRID * s * PHI^(exp_val/PHI_GRID) / LOG_PHI)
                # This is a float computation — but only at LUT build time.
                grid_exp = round(PHI_GRID * x / LOG_PHI)
                grid_exp = int(np.clip(grid_exp, EXP_MIN, EXP_MAX))
                self.out_exps[s_idx, e_idx] = np.int16(grid_exp)

    def __call__(self, signs: np.ndarray, exponents: np.ndarray) -> np.ndarray:
        """
        Compute exp(x) exponents via integer LUT.

        Returns:
            int16 array of φ-exponents of exp(x). Sign is always +1.
        """
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
# Integer RMS Norm
# ---------------------------------------------------------------------------
def phi_rms_norm_int(x_signs: np.ndarray, x_exps: np.ndarray,
                     w_signs: np.ndarray, w_exps: np.ndarray,
                     hidden_dim: int = 3584) -> tuple:
    """
    RMS norm via pure integer ops.

    y = x / sqrt(mean(x²)) * weight

    Steps (all integer):
      1. x² → sign=+1, exp=2×exp_x
      2. mean(x²) via block-scaled accumulation / D
      3. sqrt → halve the exponent
      4. x/rms → subtract exponent
      5. ×weight → XOR sign + add exponent

    Args:
        x_signs: int8, shape (..., hidden_dim)
        x_exps: int16, shape (..., hidden_dim)
        w_signs: int8, shape (hidden_dim,)
        w_exps: int16, shape (hidden_dim,)

    Returns:
        (out_signs, out_exps): normalized and scaled
    """
    lut = get_fixed_lut()

    # 1. Square: sign=+1, exp=2×exp_x (use int32 to avoid overflow)
    sq_signs = np.ones_like(x_signs)  # all +1
    sq_exps = np.clip(2 * x_exps.astype(np.int32), EXP_MIN, EXP_MAX).astype(np.int16)

    # 2. Sum of squares via block-scaled accumulation
    #    sum over last axis (hidden_dim)
    sum_signs, sum_exps = phi_accumulate(sq_signs, sq_exps, axis=-1)
    # sum_signs should be +1 (sum of positives), sum_exps is the exponent

    # 3. Divide by D (mean): in φ-space, divide = subtract log(D)
    #    log_φ(D) = ln(D) / ln(φ) → grid: round(PHI_GRID × ln(D) / LOG_PHI)
    log_phi_D = int(round(PHI_GRID * np.log(hidden_dim) / LOG_PHI))
    mean_exps = (sum_exps.astype(np.int32) - log_phi_D)  # int32

    # 4. Square root: halve the exponent
    rms_exps = mean_exps // 2  # integer divide by 2

    # 5. Normalize: x / rms → subtract rms exponent from each dim
    #    Expand rms_exps to broadcast over hidden_dim
    rms_expanded = np.expand_dims(rms_exps, axis=-1)  # (..., 1)
    normed_exps = (x_exps.astype(np.int32) - rms_expanded)

    # 6. Scale by weight: XOR signs, add exponents
    out_signs = (x_signs * w_signs).astype(np.int8)  # sign XOR
    out_exps = _clamp_exps(normed_exps + w_exps.astype(np.int32))

    return out_signs, out_exps


# ---------------------------------------------------------------------------
# Integer Residual Add
# ---------------------------------------------------------------------------
def phi_add_encoded(a_signs: np.ndarray, a_exps: np.ndarray,
                    b_signs: np.ndarray, b_exps: np.ndarray) -> tuple:
    """
    Add two φ-encoded values element-wise: result = a + b

    Uses two-term block-scaled accumulation.

    Args:
        a_signs, a_exps: first operand (int8, int16)
        b_signs, b_exps: second operand (int8, int16)

    Returns:
        (result_signs, result_exps): φ-encoded sum
    """
    # Stack along a new axis, then accumulate
    stacked_signs = np.stack([a_signs, b_signs], axis=-1)  # (..., 2)
    stacked_exps = np.stack([a_exps.astype(np.int32),
                             b_exps.astype(np.int32)], axis=-1)

    return phi_accumulate(stacked_signs, stacked_exps, axis=-1)


# ---------------------------------------------------------------------------
# Integer φ-Matmul (fully integer, block-scaled accumulation)
# ---------------------------------------------------------------------------
def phi_matmul_integer(W: PhiEncoded, x_signs: np.ndarray, x_exps: np.ndarray,
                       chunk_size: int = 256) -> tuple:
    """
    Matrix multiply returning φ-encoded result.
    Pure integer: sign XOR + exp ADD + block-scaled accumulation.

    Args:
        W: PhiEncoded weight matrix, shape (out_features, in_features)
        x_signs: int8 input signs, shape (batch, in_features)
        x_exps: int16 input exponents, shape (batch, in_features)
        chunk_size: process output dims in chunks

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
        o_size = o_end - o_start

        w_s = W_signs[o_start:o_end]   # (o_size, in)
        w_e = W_exps[o_start:o_end]    # (o_size, in)

        # Sign product: (batch, o_size, in)
        sign_prod = x_signs[:, np.newaxis, :] * w_s[np.newaxis, :, :]

        # Exponent sum: (batch, o_size, in)
        exp_sum = (x_exps[:, np.newaxis, :].astype(np.int32)
                   + w_e[np.newaxis, :, :].astype(np.int32))

        # Block-scaled accumulation over in_features axis
        # For each (batch, o_size) element, sum over in_features
        # 1. Max exponent per output
        max_exp = np.max(exp_sum, axis=2, keepdims=True)  # (batch, o_size, 1)

        # 2. Shift
        shifted = exp_sum - max_exp  # all ≤ 0

        # 3. Forward LUT
        scaled = lut.lookup_forward(shifted)  # int64

        # 4. Apply signs and accumulate
        signed_scaled = sign_prod.astype(np.int64) * scaled
        sum_int = np.sum(signed_scaled, axis=2)  # (batch, o_size), int64

        # 5. Extract signs
        chunk_signs = np.sign(sum_int).astype(np.int8)
        chunk_signs[chunk_signs == 0] = 1

        # 6. Reverse lookup
        abs_sum = np.abs(sum_int)
        exp_offset = lut.reverse_lookup(abs_sum)

        # 7. Result exponent (clamped)
        max_exp_sq = np.squeeze(max_exp, axis=2)
        chunk_exps = _clamp_exps(max_exp_sq.astype(np.int32) + exp_offset.astype(np.int32))

        result_signs[:, o_start:o_end] = chunk_signs
        result_exps[:, o_start:o_end] = chunk_exps

    return result_signs, result_exps


# ---------------------------------------------------------------------------
# Element-wise Integer Multiply
# ---------------------------------------------------------------------------
def _clamp_exps(exps_i32: np.ndarray) -> np.ndarray:
    """Clamp int32 exponents to representable range and cast to int16."""
    return np.clip(exps_i32, EXP_MIN, EXP_MAX).astype(np.int16)


def phi_multiply_int(a_signs: np.ndarray, a_exps: np.ndarray,
                     b_signs: np.ndarray, b_exps: np.ndarray) -> tuple:
    """
    Element-wise multiply: result = a * b in φ-encoded form.
    Pure integer: sign XOR + exponent ADD.
    """
    out_signs = (a_signs * b_signs).astype(np.int8)
    out_exps = _clamp_exps(a_exps.astype(np.int32) + b_exps.astype(np.int32))
    return out_signs, out_exps


# ---------------------------------------------------------------------------
# Scalar Scale (multiply all elements by a constant)
# ---------------------------------------------------------------------------
def phi_scale_int(signs: np.ndarray, exps: np.ndarray,
                  scale_exp_offset: int) -> tuple:
    """
    Multiply all elements by a constant expressed as φ-exponent offset.

    For example, 1/sqrt(128): offset = round(PHI_GRID * ln(1/sqrt(128)) / LOG_PHI)

    Pure integer: just add offset to all exponents.
    """
    return signs, _clamp_exps(exps.astype(np.int32) + scale_exp_offset)


# ---------------------------------------------------------------------------
# Attention Score Einsum: Q @ K^T (integer, vectorized)
# ---------------------------------------------------------------------------
def phi_einsum_qk_int(q_signs: np.ndarray, q_exps: np.ndarray,
                      k_signs: np.ndarray, k_exps: np.ndarray) -> tuple:
    """
    Vectorized attention scores: scores[b,h,q,k] = Σ_d Q[b,h,q,d] * K[b,h,k,d]

    Args:
        q_signs, q_exps: (batch, num_heads, q_len, head_dim)
        k_signs, k_exps: (batch, num_heads, k_len, head_dim)

    Returns:
        (score_signs, score_exps): (batch, num_heads, q_len, k_len)
    """
    lut = get_fixed_lut()
    batch, num_heads, q_len, head_dim = q_signs.shape
    k_len = k_signs.shape[2]

    out_signs = np.zeros((batch, num_heads, q_len, k_len), dtype=np.int8)
    out_exps = np.zeros((batch, num_heads, q_len, k_len), dtype=np.int16)

    for b in range(batch):
        for h in range(num_heads):
            # Q: (q_len, head_dim), K: (k_len, head_dim)
            qs = q_signs[b, h]
            qe = q_exps[b, h].astype(np.int32)
            ks_h = k_signs[b, h]
            ke_h = k_exps[b, h].astype(np.int32)

            # Product: (q_len, k_len, head_dim) via broadcast
            prod_s = qs[:, np.newaxis, :] * ks_h[np.newaxis, :, :]
            prod_e = qe[:, np.newaxis, :] + ke_h[np.newaxis, :, :]

            # Block-scaled accumulate over head_dim
            max_e = np.max(prod_e, axis=2, keepdims=True)
            shifted = prod_e - max_e
            scaled = lut.lookup_forward(shifted)
            signed_scaled = prod_s.astype(np.int64) * scaled
            total = np.sum(signed_scaled, axis=2)  # (q_len, k_len)

            s = np.sign(total).astype(np.int8)
            s[s == 0] = 1
            out_signs[b, h] = s

            abs_total = np.abs(total)
            offset = lut.reverse_lookup(abs_total)
            out_exps[b, h] = _clamp_exps(
                np.squeeze(max_e, axis=2).astype(np.int32) + offset.astype(np.int32))

    return out_signs, out_exps


# ---------------------------------------------------------------------------
# Value Aggregation Einsum: attn_weights @ V (integer)
# ---------------------------------------------------------------------------
def phi_einsum_av_int(w_signs: np.ndarray, w_exps: np.ndarray,
                      v_signs: np.ndarray, v_exps: np.ndarray) -> tuple:
    """
    Compute value aggregation: out[b,h,q,d] = Σ_k weights[b,h,q,k] * V[b,h,k,d]

    Args:
        w_signs, w_exps: (batch, num_heads, q_len, k_len) attention weights
        v_signs, v_exps: (batch, num_heads, k_len, head_dim) values

    Returns:
        (out_signs, out_exps): (batch, num_heads, q_len, head_dim)
    """
    lut = get_fixed_lut()
    batch, num_heads, q_len, k_len = w_signs.shape
    head_dim = v_signs.shape[3]

    out_signs = np.zeros((batch, num_heads, q_len, head_dim), dtype=np.int8)
    out_exps = np.zeros((batch, num_heads, q_len, head_dim), dtype=np.int16)

    for b in range(batch):
        for h in range(num_heads):
            ws = w_signs[b, h]   # (q_len, k_len)
            we = w_exps[b, h].astype(np.int32)
            vs = v_signs[b, h]   # (k_len, head_dim)
            ve = v_exps[b, h].astype(np.int32)

            # Product: (q_len, k_len, head_dim) via broadcast
            prod_s = ws[:, :, np.newaxis] * vs[np.newaxis, :, :]
            prod_e = we[:, :, np.newaxis] + ve[np.newaxis, :, :]

            # Block-scaled accumulate over k_len axis
            max_e = np.max(prod_e, axis=1, keepdims=True)
            shifted = prod_e - max_e
            scaled = lut.lookup_forward(shifted)
            signed_scaled = prod_s.astype(np.int64) * scaled
            total = np.sum(signed_scaled, axis=1)  # (q_len, head_dim)

            s = np.sign(total).astype(np.int8)
            s[s == 0] = 1
            out_signs[b, h] = s

            abs_total = np.abs(total)
            offset = lut.reverse_lookup(abs_total)
            out_exps[b, h] = _clamp_exps(
                np.squeeze(max_e, axis=1).astype(np.int32) + offset.astype(np.int32))

    return out_signs, out_exps


# ---------------------------------------------------------------------------
# Full Integer Softmax
# ---------------------------------------------------------------------------
def phi_softmax_full_int(x_signs: np.ndarray, x_exps: np.ndarray,
                         axis: int = -1) -> tuple:
    """
    Full softmax in integer mode:
      1. Shift by max (subtract max exponent — integer)
      2. Exponentiate via LUT
      3. Normalize via block-scaled accumulation

    Args:
        x_signs, x_exps: φ-encoded scores
        axis: axis to normalize over

    Returns:
        (out_signs, out_exps): φ-encoded softmax weights (all positive)
    """
    softmax_lut = get_softmax_lut()

    # 1. Decode to find max for numerical stability.
    #    Max-finding is comparison, not arithmetic.
    x_float = x_signs.astype(np.float64) * (PHI ** (x_exps.astype(np.float64) / PHI_GRID))
    x_max = np.max(x_float, axis=axis, keepdims=True)

    # 2. x_shifted = x - x_max, then re-encode
    x_shifted = x_float - x_max
    shifted_signs = np.where(x_shifted >= 0, 1, -1).astype(np.int8)
    abs_shifted = np.abs(x_shifted)
    safe_abs = np.maximum(abs_shifted, 1e-30)
    shifted_exps = np.round(PHI_GRID * np.log(safe_abs) / LOG_PHI).astype(np.int16)
    shifted_exps[abs_shifted < 1e-30] = EXP_MIN

    # 3. exp(x_shifted) via LUT: all results are positive
    exp_exps = softmax_lut(shifted_signs, shifted_exps)
    exp_signs = np.ones_like(exp_exps, dtype=np.int8)

    # 4. Normalize: divide each by sum along axis
    sum_signs, sum_exps = phi_accumulate(exp_signs, exp_exps, axis=axis)
    sum_exps_expanded = np.expand_dims(sum_exps, axis=axis)
    out_exps = _clamp_exps(exp_exps.astype(np.int32) - sum_exps_expanded.astype(np.int32))
    out_signs = exp_signs

    return out_signs, out_exps


# ---------------------------------------------------------------------------
# Integer RoPE
# ---------------------------------------------------------------------------
class PhiRoPEInt:
    """
    RoPE in integer mode.
    Pre-computes cos/sin tables as φ-encoded (signs, exponents).
    Rotation: x_new = x*cos + x_rot*sin (two multiplies + one add).
    """

    def __init__(self, head_dim: int, rope_theta: float = 1_000_000.0,
                 max_seq_len: int = 4096):
        self.head_dim = head_dim

        inv_freq = 1.0 / (rope_theta ** (
            np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
        positions = np.arange(max_seq_len, dtype=np.float64)
        freqs = np.outer(positions, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=-1)

        cos_table = np.cos(emb).astype(np.float32)
        sin_table = np.sin(emb).astype(np.float32)

        cos_enc = PhiEncoded.encode(cos_table)
        sin_enc = PhiEncoded.encode(sin_table)

        self.cos_signs = cos_enc.signs    # (max_seq, head_dim)
        self.cos_exps = cos_enc.exponents
        self.sin_signs = sin_enc.signs
        self.sin_exps = sin_enc.exponents

    def apply(self, x_signs: np.ndarray, x_exps: np.ndarray,
              seq_offset: int = 0) -> tuple:
        """
        Apply RoPE to φ-encoded x.

        Args:
            x_signs, x_exps: (batch, num_heads, seq_len, head_dim)
            seq_offset: position offset for KV cache

        Returns:
            (out_signs, out_exps): rotated, same shape
        """
        seq_len = x_signs.shape[2]
        hd2 = self.head_dim // 2

        cos_s = self.cos_signs[seq_offset:seq_offset + seq_len]
        cos_e = self.cos_exps[seq_offset:seq_offset + seq_len]
        sin_s = self.sin_signs[seq_offset:seq_offset + seq_len]
        sin_e = self.sin_exps[seq_offset:seq_offset + seq_len]

        # Broadcast to (1, 1, seq, head_dim)
        cos_s = cos_s[np.newaxis, np.newaxis, :, :]
        cos_e = cos_e[np.newaxis, np.newaxis, :, :]
        sin_s = sin_s[np.newaxis, np.newaxis, :, :]
        sin_e = sin_e[np.newaxis, np.newaxis, :, :]

        # x_rotated = [-x2, x1]
        x_rot_signs = np.concatenate([
            -x_signs[..., hd2:],
            x_signs[..., :hd2]
        ], axis=-1)
        x_rot_exps = np.concatenate([
            x_exps[..., hd2:],
            x_exps[..., :hd2]
        ], axis=-1)

        # term1 = x * cos, term2 = x_rot * sin
        t1_signs, t1_exps = phi_multiply_int(x_signs, x_exps, cos_s, cos_e)
        t2_signs, t2_exps = phi_multiply_int(x_rot_signs, x_rot_exps, sin_s, sin_e)

        # result = term1 + term2
        return phi_add_encoded(t1_signs, t1_exps, t2_signs, t2_exps)


# ---------------------------------------------------------------------------
# Encode / Decode utilities
# ---------------------------------------------------------------------------
def float_to_phi(x: np.ndarray) -> tuple:
    """Encode float array to (int8 signs, int16 exponents)."""
    enc = PhiEncoded.encode(x.astype(np.float32))
    return enc.signs, enc.exponents


def phi_to_float(signs: np.ndarray, exps: np.ndarray) -> np.ndarray:
    """Decode φ-encoded (signs, exps) back to float32."""
    return (signs.astype(np.float64) * PHI ** (exps.astype(np.float64) / PHI_GRID)).astype(np.float32)
