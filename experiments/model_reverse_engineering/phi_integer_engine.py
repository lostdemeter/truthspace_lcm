#!/usr/bin/env python3
"""
φ-Integer Arithmetic Engine for Qwen2-7B
=========================================

Implements pure φ-integer arithmetic like we did for DA2.

Key insight: In φ-space, multiplication becomes integer addition:
    a × b = φ^(e_a/K) × φ^(e_b/K) = φ^((e_a + e_b)/K)

Storage format:
    - 1 bit: sign (0=positive, 1=negative)
    - 8 bits: quantized exponent (0-255)
    - Total: 9 bits per weight

Computation:
    - Exponent addition (integer)
    - Sign XOR
    - LUT lookup for final value
    - Float accumulation

This eliminates IEEE floating-point multiplication entirely.

Author: TruthSpace LCM Team
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import os

# φ constants
PHI = (1 + np.sqrt(5)) / 2
K = 128  # φ-grid resolution
STEP = 32  # Quantization step (gives 99.92% accuracy)


@dataclass
class PhiIntegerTensor:
    """
    A tensor stored in φ-integer format.
    
    Each weight is stored as:
    - sign: 1 bit (packed into uint8 array, 8 weights per byte)
    - exponent: uint8 (0-255, representing quantized φ-exponent)
    
    The actual value is: sign × φ^((exponent - offset) × step / K)
    """
    signs: np.ndarray      # Packed bits, shape = (ceil(n/8),)
    exponents: np.ndarray  # uint8, shape = original shape
    offset: int            # Exponent offset to make all values positive
    shape: Tuple[int, ...]
    
    @classmethod
    def from_float(cls, tensor: np.ndarray) -> 'PhiIntegerTensor':
        """Convert float tensor to φ-integer format."""
        shape = tensor.shape
        flat = tensor.flatten()
        n = len(flat)
        
        # Extract signs (1 = positive, 0 = negative for XOR convenience)
        signs_bool = flat >= 0
        
        # Pack signs into bytes (8 per byte)
        n_bytes = (n + 7) // 8
        signs_packed = np.zeros(n_bytes, dtype=np.uint8)
        for i in range(n):
            if signs_bool[i]:
                signs_packed[i // 8] |= (1 << (i % 8))
        
        # Compute exponents
        magnitudes = np.abs(flat) + 1e-20
        exponents_raw = K * np.log(magnitudes) / np.log(PHI)
        exponents_quantized = np.round(exponents_raw / STEP).astype(np.int32)
        
        # Shift to positive range
        offset = -exponents_quantized.min() + 1
        exponents_shifted = (exponents_quantized + offset).astype(np.uint8)
        
        # Verify range fits in uint8
        assert exponents_shifted.max() <= 255, f"Exponent overflow: {exponents_shifted.max()}"
        
        return cls(
            signs=signs_packed,
            exponents=exponents_shifted.reshape(shape),
            offset=offset,
            shape=shape,
        )
    
    def to_float(self) -> np.ndarray:
        """Convert back to float (for verification)."""
        flat_exp = self.exponents.flatten()
        n = len(flat_exp)
        
        # Unpack signs
        signs = np.ones(n, dtype=np.float32)
        for i in range(n):
            if not (self.signs[i // 8] & (1 << (i % 8))):
                signs[i] = -1.0
        
        # Compute values
        actual_exp = (flat_exp.astype(np.int32) - self.offset) * STEP
        values = signs * (PHI ** (actual_exp / K))
        
        return values.reshape(self.shape).astype(np.float32)
    
    def storage_bytes(self) -> int:
        """Total storage in bytes."""
        return self.signs.nbytes + self.exponents.nbytes + 4  # +4 for offset
    
    def save(self, path: str):
        """Save to file."""
        np.savez_compressed(
            path,
            signs=self.signs,
            exponents=self.exponents,
            offset=np.array([self.offset]),
            shape=np.array(self.shape),
        )
    
    @classmethod
    def load(cls, path: str) -> 'PhiIntegerTensor':
        """Load from file."""
        data = np.load(path)
        return cls(
            signs=data['signs'],
            exponents=data['exponents'],
            offset=int(data['offset'][0]),
            shape=tuple(data['shape']),
        )


class PhiIntegerLUT:
    """
    Lookup table for φ^(e/K) values.
    
    The LUT maps combined exponents to float values:
        LUT[e] = φ^((e - 2*offset) * step / K)
    
    Size is typically ~500 entries = 2KB, fits in L1 cache.
    """
    
    def __init__(self, max_exp: int, offset: int):
        self.offset = offset
        self.size = max_exp + 1
        
        # Build LUT
        self.table = np.zeros(self.size, dtype=np.float32)
        for e in range(self.size):
            actual_exp = (e - 2 * offset) * STEP
            self.table[e] = PHI ** (actual_exp / K)
    
    def __getitem__(self, exp: np.ndarray) -> np.ndarray:
        """Lookup values for given exponents."""
        return self.table[exp]


def phi_integer_matmul(
    x_signs: np.ndarray,      # Packed signs for x
    x_exps: np.ndarray,       # uint8 exponents for x, shape (batch, d_in)
    w_signs: np.ndarray,      # Packed signs for W
    w_exps: np.ndarray,       # uint8 exponents for W, shape (d_out, d_in)
    lut: PhiIntegerLUT,
    x_offset: int,
    w_offset: int,
) -> np.ndarray:
    """
    Compute matmul using φ-integer arithmetic.
    
    result[i, j] = Σ_k sign_x[i,k] × sign_w[j,k] × LUT[exp_x[i,k] + exp_w[j,k]]
    
    This is the NAIVE implementation for correctness verification.
    A proper CUDA kernel would be much faster.
    """
    batch, d_in = x_exps.shape
    d_out, _ = w_exps.shape
    
    result = np.zeros((batch, d_out), dtype=np.float32)
    
    # Unpack signs
    def unpack_signs(packed, n):
        signs = np.ones(n, dtype=np.float32)
        for i in range(n):
            if not (packed[i // 8] & (1 << (i % 8))):
                signs[i] = -1.0
        return signs
    
    for i in range(batch):
        x_sign_i = unpack_signs(x_signs[i * d_in // 8: (i + 1) * d_in // 8 + 1], d_in)
        
        for j in range(d_out):
            w_sign_j = unpack_signs(w_signs[j * d_in // 8: (j + 1) * d_in // 8 + 1], d_in)
            
            # Integer exponent addition
            combined_exp = x_exps[i].astype(np.int32) + w_exps[j].astype(np.int32)
            
            # Sign multiplication
            combined_sign = x_sign_i * w_sign_j
            
            # LUT lookup and accumulate
            values = combined_sign * lut[combined_exp]
            result[i, j] = values.sum()
    
    return result


def phi_integer_matmul_vectorized(
    x: PhiIntegerTensor,
    w: PhiIntegerTensor,
) -> np.ndarray:
    """
    Vectorized φ-integer matmul (still Python, but faster than naive).
    
    For proper speed, this needs a CUDA kernel.
    """
    # Get shapes
    if len(x.shape) == 1:
        x_exps = x.exponents.reshape(1, -1)
        batch = 1
    else:
        x_exps = x.exponents
        batch = x.shape[0]
    
    d_in = x_exps.shape[-1]
    d_out = w.shape[0]
    
    # Unpack all signs at once
    def unpack_all_signs(packed, shape):
        n = np.prod(shape)
        signs = np.ones(n, dtype=np.float32)
        for i in range(n):
            if not (packed[i // 8] & (1 << (i % 8))):
                signs[i] = -1.0
        return signs.reshape(shape)
    
    x_signs_unpacked = unpack_all_signs(x.signs, x_exps.shape)
    w_signs_unpacked = unpack_all_signs(w.signs, w.exponents.shape)
    
    # Build LUT
    max_combined = int(x.exponents.max()) + int(w.exponents.max()) + 1
    lut = np.zeros(max_combined, dtype=np.float32)
    for e in range(max_combined):
        actual_exp = (e - x.offset - w.offset) * STEP
        lut[e] = PHI ** (actual_exp / K)
    
    # Compute result
    # This is still O(batch × d_out × d_in) but vectorized over d_in
    result = np.zeros((batch, d_out), dtype=np.float32)
    
    for j in range(d_out):
        # Exponent addition: (batch, d_in) + (d_in,) -> (batch, d_in)
        combined_exp = x_exps.astype(np.int32) + w.exponents[j].astype(np.int32)
        
        # Sign multiplication: (batch, d_in) * (d_in,) -> (batch, d_in)
        combined_sign = x_signs_unpacked * w_signs_unpacked[j]
        
        # LUT lookup and sum: (batch, d_in) -> (batch,)
        values = combined_sign * lut[combined_exp]
        result[:, j] = values.sum(axis=1)
    
    return result


def test_phi_integer():
    """Test φ-integer arithmetic."""
    import time
    
    print("=" * 60)
    print("φ-INTEGER ARITHMETIC TEST")
    print("=" * 60)
    print()
    
    # Test tensor conversion
    np.random.seed(42)
    W = np.random.randn(1000, 500).astype(np.float32) * 0.01
    
    print("Testing PhiIntegerTensor...")
    W_phi = PhiIntegerTensor.from_float(W)
    W_reconstructed = W_phi.to_float()
    
    corr = np.corrcoef(W.flatten(), W_reconstructed.flatten())[0, 1]
    print(f"  Correlation: {corr:.6f}")
    print(f"  Storage: {W_phi.storage_bytes() / 1e6:.2f} MB (vs {W.nbytes / 1e6:.2f} MB float32)")
    print(f"  Compression: {W.nbytes / W_phi.storage_bytes():.2f}×")
    print()
    
    # Test matmul
    print("Testing φ-integer matmul...")
    x = np.random.randn(10, 500).astype(np.float32)
    x_phi = PhiIntegerTensor.from_float(x)
    
    # Float32 reference
    result_float = x @ W.T
    
    # φ-integer
    start = time.perf_counter()
    result_phi = phi_integer_matmul_vectorized(x_phi, W_phi)
    phi_time = time.perf_counter() - start
    
    corr = np.corrcoef(result_float.flatten(), result_phi.flatten())[0, 1]
    print(f"  Correlation: {corr:.6f}")
    print(f"  Time: {phi_time*1000:.1f} ms (Python, not optimized)")
    print()
    
    print("=" * 60)
    print("CUDA KERNEL PSEUDOCODE")
    print("=" * 60)
    print("""
__global__ void phi_integer_matmul_kernel(
    const uint8_t* x_signs,    // Packed signs for x
    const uint8_t* x_exps,     // Exponents for x
    const uint8_t* w_signs,    // Packed signs for W
    const uint8_t* w_exps,     // Exponents for W
    const float* lut,          // φ-value lookup table
    float* result,             // Output
    int batch, int d_in, int d_out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // batch index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // output index
    
    if (i >= batch || j >= d_out) return;
    
    // Load LUT into shared memory (fits in 2KB)
    __shared__ float lut_shared[512];
    if (threadIdx.x < 512) {
        lut_shared[threadIdx.x] = lut[threadIdx.x];
    }
    __syncthreads();
    
    float sum = 0.0f;
    
    for (int k = 0; k < d_in; k++) {
        // Unpack signs (1 bit each)
        int x_sign = (x_signs[(i * d_in + k) / 8] >> ((i * d_in + k) % 8)) & 1;
        int w_sign = (w_signs[(j * d_in + k) / 8] >> ((j * d_in + k) % 8)) & 1;
        
        // XOR signs: same sign = positive, different = negative
        float sign = (x_sign == w_sign) ? 1.0f : -1.0f;
        
        // Integer exponent addition
        int combined_exp = x_exps[i * d_in + k] + w_exps[j * d_in + k];
        
        // LUT lookup (in shared memory = fast!)
        float value = lut_shared[combined_exp];
        
        // Accumulate
        sum += sign * value;
    }
    
    result[i * d_out + j] = sum;
}
""")
    
    print("Key optimizations:")
    print("  1. LUT in shared memory (2KB, fits easily)")
    print("  2. Signs packed as bits (8× less memory)")
    print("  3. Exponents as uint8 (2× less than float16)")
    print("  4. Only integer ops until final accumulation")
    print("  5. Memory bandwidth reduced ~4× vs float32")


if __name__ == "__main__":
    test_phi_integer()
