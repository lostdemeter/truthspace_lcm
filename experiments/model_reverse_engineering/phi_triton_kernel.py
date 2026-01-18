#!/usr/bin/env python3
"""
φ-Integer Matmul CUDA Kernel using Triton
==========================================

Implements φ-arithmetic matmul where multiplication becomes integer addition:
    a × b = φ^(e_a/K) × φ^(e_b/K) = φ^((e_a + e_b)/K)

Storage format:
    - Exponents: uint8 (0-255)
    - Signs: packed bits (8 per byte)
    
Computation:
    result[i,j] = Σ_k sign[i,k] × sign[j,k] × LUT[exp_x[i,k] + exp_w[j,k]]

The LUT is small (~2KB) and fits in L1 cache / shared memory.

Author: TruthSpace LCM Team
"""

import torch
import triton
import triton.language as tl
import numpy as np
import time

# φ constants
PHI = (1 + np.sqrt(5)) / 2
K = 128
STEP = 32


@triton.jit
def phi_matmul_kernel(
    # Pointers to matrices
    x_exp_ptr, x_sign_ptr,  # Input: exponents and packed signs
    w_exp_ptr, w_sign_ptr,  # Weights: exponents and packed signs
    lut_ptr,                 # LUT for φ^(e/K)
    out_ptr,                 # Output
    # Matrix dimensions
    M, N, K_dim,            # M = batch, N = d_out, K_dim = d_in
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    # LUT offset
    lut_offset,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    φ-integer matmul kernel.
    
    For each output element (m, n):
        out[m, n] = Σ_k sign_x[m,k] × sign_w[n,k] × LUT[exp_x[m,k] + exp_w[n,k]]
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k_start in range(0, K_dim, BLOCK_K):
        k_offs = k_start + offs_k
        
        # Load x exponents: (BLOCK_M, BLOCK_K)
        x_exp_ptrs = x_exp_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_exp = tl.load(x_exp_ptrs, mask=(offs_m[:, None] < M) & (k_offs[None, :] < K_dim), other=0)
        
        # Load w exponents: (BLOCK_N, BLOCK_K)
        w_exp_ptrs = w_exp_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
        w_exp = tl.load(w_exp_ptrs, mask=(offs_n[:, None] < N) & (k_offs[None, :] < K_dim), other=0)
        
        # Load x signs (packed bytes)
        # Each byte contains 8 signs, so we need byte index and bit position
        x_byte_idx = (offs_m[:, None] * K_dim + k_offs[None, :]) // 8
        x_bit_pos = (offs_m[:, None] * K_dim + k_offs[None, :]) % 8
        x_sign_bytes = tl.load(x_sign_ptr + x_byte_idx, 
                               mask=(offs_m[:, None] < M) & (k_offs[None, :] < K_dim), other=0)
        x_sign = ((x_sign_bytes >> x_bit_pos) & 1)  # 0 or 1
        x_sign_float = tl.where(x_sign == 1, 1.0, -1.0)  # 1 -> +1, 0 -> -1
        
        # Load w signs (packed bytes)
        w_byte_idx = (offs_n[:, None] * K_dim + k_offs[None, :]) // 8
        w_bit_pos = (offs_n[:, None] * K_dim + k_offs[None, :]) % 8
        w_sign_bytes = tl.load(w_sign_ptr + w_byte_idx,
                               mask=(offs_n[:, None] < N) & (k_offs[None, :] < K_dim), other=0)
        w_sign = ((w_sign_bytes >> w_bit_pos) & 1)
        w_sign_float = tl.where(w_sign == 1, 1.0, -1.0)
        
        # Combined exponent: (BLOCK_M, BLOCK_K) + (BLOCK_N, BLOCK_K) via broadcast
        # We need (BLOCK_M, BLOCK_N, BLOCK_K) but that's too much memory
        # Instead, loop over BLOCK_K and accumulate
        
        for kk in range(BLOCK_K):
            if k_start + kk < K_dim:
                # Get exponents for this k
                x_e = tl.load(x_exp_ptr + offs_m * stride_xm + (k_start + kk) * stride_xk,
                             mask=offs_m < M, other=0)
                w_e = tl.load(w_exp_ptr + offs_n * stride_wn + (k_start + kk) * stride_wk,
                             mask=offs_n < N, other=0)
                
                # Combined exponent: (BLOCK_M,) + (BLOCK_N,) -> (BLOCK_M, BLOCK_N)
                combined_exp = x_e[:, None] + w_e[None, :]
                combined_exp = combined_exp.to(tl.int32) + lut_offset
                
                # LUT lookup
                lut_vals = tl.load(lut_ptr + combined_exp)
                
                # Get signs for this k
                x_byte = (offs_m * K_dim + k_start + kk) // 8
                x_bit = (offs_m * K_dim + k_start + kk) % 8
                x_s_byte = tl.load(x_sign_ptr + x_byte, mask=offs_m < M, other=0)
                x_s = ((x_s_byte >> x_bit) & 1)
                x_s_f = tl.where(x_s == 1, 1.0, -1.0)
                
                w_byte = (offs_n * K_dim + k_start + kk) // 8
                w_bit = (offs_n * K_dim + k_start + kk) % 8
                w_s_byte = tl.load(w_sign_ptr + w_byte, mask=offs_n < N, other=0)
                w_s = ((w_s_byte >> w_bit) & 1)
                w_s_f = tl.where(w_s == 1, 1.0, -1.0)
                
                # Combined sign: (BLOCK_M,) * (BLOCK_N,) -> (BLOCK_M, BLOCK_N)
                combined_sign = x_s_f[:, None] * w_s_f[None, :]
                
                # Accumulate
                acc += combined_sign * lut_vals
    
    # Store result
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def phi_matmul_triton(
    x_exp: torch.Tensor,    # (M, K) uint8
    x_sign: torch.Tensor,   # Packed signs
    w_exp: torch.Tensor,    # (N, K) uint8
    w_sign: torch.Tensor,   # Packed signs
    lut: torch.Tensor,      # LUT for φ values
    lut_offset: int,
) -> torch.Tensor:
    """
    φ-integer matmul using Triton kernel.
    
    Args:
        x_exp: Input exponents, shape (M, K), dtype uint8
        x_sign: Packed input signs, shape (ceil(M*K/8),), dtype uint8
        w_exp: Weight exponents, shape (N, K), dtype uint8
        w_sign: Packed weight signs, shape (ceil(N*K/8),), dtype uint8
        lut: Lookup table for φ^(e/K), shape (lut_size,), dtype float32
        lut_offset: Offset to add to combined exponents before LUT lookup
        
    Returns:
        Output tensor, shape (M, N), dtype float32
    """
    M, K_dim = x_exp.shape
    N, _ = w_exp.shape
    
    # Allocate output
    out = torch.zeros((M, N), dtype=torch.float32, device=x_exp.device)
    
    # Grid
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Launch kernel
    phi_matmul_kernel[grid](
        x_exp, x_sign,
        w_exp, w_sign,
        lut,
        out,
        M, N, K_dim,
        x_exp.stride(0), x_exp.stride(1),
        w_exp.stride(0), w_exp.stride(1),
        out.stride(0), out.stride(1),
        lut_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return out


# Optimized kernel - unpacked signs for simplicity
@triton.jit
def phi_matmul_v2_kernel(
    x_exp_ptr,   # (M, K) uint8 exponents
    x_sign_ptr,  # (M, K) int8 signs (+1 or -1)
    w_exp_ptr,   # (N, K) uint8 exponents
    w_sign_ptr,  # (N, K) int8 signs (+1 or -1)
    lut_ptr,     # LUT for φ values
    out_ptr,     # (M, N) output
    M, N, K_dim,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    φ-matmul kernel with unpacked signs for simplicity.
    
    Uses tiled computation for good GPU utilization.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K_dim, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # Load x exponents: (BLOCK_M, BLOCK_K)
        x_exp_ptrs = x_exp_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_exp = tl.load(x_exp_ptrs, 
                        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K_dim),
                        other=0)
        
        # Load x signs: (BLOCK_M, BLOCK_K)
        x_sign_ptrs = x_sign_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_sign = tl.load(x_sign_ptrs,
                         mask=(offs_m[:, None] < M) & (offs_k[None, :] < K_dim),
                         other=1).to(tl.float32)
        
        # Load w exponents: (BLOCK_N, BLOCK_K)
        w_exp_ptrs = w_exp_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_exp = tl.load(w_exp_ptrs,
                        mask=(offs_n[:, None] < N) & (offs_k[None, :] < K_dim),
                        other=0)
        
        # Load w signs: (BLOCK_N, BLOCK_K)
        w_sign_ptrs = w_sign_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_sign = tl.load(w_sign_ptrs,
                         mask=(offs_n[:, None] < N) & (offs_k[None, :] < K_dim),
                         other=1).to(tl.float32)
        
        # Compute contribution for this K block
        # We need: acc[m,n] += sum_k sign_x[m,k] * sign_w[n,k] * LUT[exp_x[m,k] + exp_w[n,k]]
        
        # For each k in the block
        for kk in tl.static_range(BLOCK_K):
            # Get column kk from each tile
            x_e = tl.load(x_exp_ptr + offs_m * stride_xm + (k + kk) * stride_xk,
                          mask=(offs_m < M) & (k + kk < K_dim), other=0)
            w_e = tl.load(w_exp_ptr + offs_n * stride_wn + (k + kk) * stride_wk,
                          mask=(offs_n < N) & (k + kk < K_dim), other=0)
            x_s = tl.load(x_sign_ptr + offs_m * stride_xm + (k + kk) * stride_xk,
                          mask=(offs_m < M) & (k + kk < K_dim), other=1).to(tl.float32)
            w_s = tl.load(w_sign_ptr + offs_n * stride_wn + (k + kk) * stride_wk,
                          mask=(offs_n < N) & (k + kk < K_dim), other=1).to(tl.float32)
            
            # Combined exponent: (BLOCK_M,) + (BLOCK_N,) -> (BLOCK_M, BLOCK_N)
            combined_exp = x_e[:, None].to(tl.int32) + w_e[None, :].to(tl.int32)
            
            # LUT lookup
            phi_vals = tl.load(lut_ptr + combined_exp)
            
            # Combined sign
            combined_sign = x_s[:, None] * w_s[None, :]
            
            # Accumulate
            acc += combined_sign * phi_vals
    
    # Store result
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def phi_matmul_v2(
    x_exp: torch.Tensor,    # (M, K) uint8
    x_sign: torch.Tensor,   # (M, K) int8 (+1 or -1)
    w_exp: torch.Tensor,    # (N, K) uint8
    w_sign: torch.Tensor,   # (N, K) int8 (+1 or -1)
    lut: torch.Tensor,      # LUT
) -> torch.Tensor:
    """φ-matmul using v2 kernel with unpacked signs."""
    M, K_dim = x_exp.shape
    N, _ = w_exp.shape
    
    out = torch.zeros((M, N), dtype=torch.float32, device=x_exp.device)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 64
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    phi_matmul_v2_kernel[grid](
        x_exp, x_sign,
        w_exp, w_sign,
        lut,
        out,
        M, N, K_dim,
        x_exp.stride(0), x_exp.stride(1),
        w_exp.stride(0), w_exp.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return out


def quantize_to_phi_v2(tensor: np.ndarray, step: int = STEP) -> tuple:
    """
    Quantize tensor to φ-integer format with unpacked signs.
    
    Returns:
        exponents: uint8 array, same shape as tensor
        signs: int8 array (+1 or -1), same shape as tensor
        offset: int (to shift exponents to positive range)
    """
    signs = np.sign(tensor).astype(np.int8)
    signs[signs == 0] = 1
    
    magnitudes = np.abs(tensor) + 1e-20
    exponents_raw = K * np.log(magnitudes) / np.log(PHI)
    exponents_quantized = np.round(exponents_raw / step).astype(np.int32)
    
    offset = -exponents_quantized.min() + 1
    exponents = (exponents_quantized + offset).astype(np.uint8)
    
    return exponents, signs, offset


def build_lut(max_exp: int, offset: int, step: int = STEP) -> np.ndarray:
    """Build LUT for φ^((e - 2*offset) * step / K)."""
    lut = np.zeros(max_exp + 1, dtype=np.float32)
    for e in range(max_exp + 1):
        actual_exp = (e - 2 * offset) * step
        lut[e] = PHI ** (actual_exp / K)
    return lut


def benchmark():
    """Benchmark φ-integer matmul vs standard float32."""
    print("=" * 60)
    print("φ-INTEGER TRITON KERNEL BENCHMARK")
    print("=" * 60)
    print()
    
    device = 'cuda'
    
    # Test dimensions (MLP-like)
    M = 512   # batch * seq_len
    K_dim = 3584   # hidden_dim
    N = 18944  # intermediate_size
    
    print(f"Matrix dimensions: ({M}, {K_dim}) @ ({N}, {K_dim}).T")
    print()
    
    # Create test data
    np.random.seed(42)
    x_float = np.random.randn(M, K_dim).astype(np.float32) * 0.1
    w_float = np.random.randn(N, K_dim).astype(np.float32) * 0.01
    
    # Quantize using v2 (unpacked signs)
    x_exp, x_sign, x_offset = quantize_to_phi_v2(x_float)
    w_exp, w_sign, w_offset = quantize_to_phi_v2(w_float)
    
    # Build LUT
    max_combined = int(x_exp.max()) + int(w_exp.max()) + 1
    lut_offset = x_offset + w_offset
    lut = build_lut(max_combined + lut_offset, lut_offset)
    
    print(f"LUT size: {len(lut)} entries ({len(lut) * 4 / 1024:.1f} KB)")
    print(f"x_exp range: [0, {x_exp.max()}]")
    print(f"w_exp range: [0, {w_exp.max()}]")
    print()
    
    # Move to GPU
    x_exp_t = torch.tensor(x_exp, dtype=torch.uint8, device=device)
    x_sign_t = torch.tensor(x_sign, dtype=torch.int8, device=device)
    w_exp_t = torch.tensor(w_exp, dtype=torch.uint8, device=device)
    w_sign_t = torch.tensor(w_sign, dtype=torch.int8, device=device)
    lut_t = torch.tensor(lut, dtype=torch.float32, device=device)
    
    x_float_t = torch.tensor(x_float, dtype=torch.float32, device=device)
    w_float_t = torch.tensor(w_float, dtype=torch.float32, device=device)
    
    # Storage comparison
    float_bytes = x_float.nbytes + w_float.nbytes
    phi_bytes = x_exp.nbytes + x_sign.nbytes + w_exp.nbytes + w_sign.nbytes + lut.nbytes
    
    print("Storage:")
    print(f"  Float32: {float_bytes / 1e6:.1f} MB")
    print(f"  φ-integer: {phi_bytes / 1e6:.1f} MB")
    print(f"  Compression: {float_bytes / phi_bytes:.2f}×")
    print()
    
    # Warmup
    _ = x_float_t @ w_float_t.T
    torch.cuda.synchronize()
    
    # Benchmark float32
    n_runs = 100
    start = time.perf_counter()
    for _ in range(n_runs):
        result_float = x_float_t @ w_float_t.T
    torch.cuda.synchronize()
    float_time = (time.perf_counter() - start) / n_runs * 1000
    
    print(f"Float32 matmul: {float_time:.2f} ms")
    
    # Benchmark φ-integer (v2 kernel with unpacked signs)
    try:
        # Warmup
        result_phi = phi_matmul_v2(x_exp_t, x_sign_t, w_exp_t, w_sign_t, lut_t)
        torch.cuda.synchronize()
        
        n_runs_phi = 100
        start = time.perf_counter()
        for _ in range(n_runs_phi):
            result_phi = phi_matmul_v2(x_exp_t, x_sign_t, w_exp_t, w_sign_t, lut_t)
        torch.cuda.synchronize()
        phi_time = (time.perf_counter() - start) / n_runs_phi * 1000
        
        print(f"φ-integer (Triton kernel): {phi_time:.2f} ms")
        print(f"Speedup: {float_time / phi_time:.2f}×")
        print()
        
        # Verify correctness
        result_float_np = result_float.cpu().numpy()
        result_phi_np = result_phi.cpu().numpy()
        
        corr = np.corrcoef(result_float_np.flatten(), result_phi_np.flatten())[0, 1]
        print(f"Correlation: {corr:.6f}")
        
    except Exception as e:
        print(f"Triton kernel error: {e}")
        print()
        print("Using vectorized PyTorch instead...")
        
        # Vectorized PyTorch reference (fast)
        def phi_matmul_pytorch_vec(x_exp, x_sign, w_exp, w_sign, lut):
            M, K = x_exp.shape
            N, _ = w_exp.shape
            
            x_sign_f = x_sign.float()
            w_sign_f = w_sign.float()
            
            # Compute in chunks to avoid OOM
            chunk_size = 256
            result = torch.zeros(M, N, dtype=torch.float32, device=x_exp.device)
            
            for j_start in range(0, N, chunk_size):
                j_end = min(j_start + chunk_size, N)
                
                # For this chunk of outputs
                for j in range(j_start, j_end):
                    combined_exp = x_exp.int() + w_exp[j].int()
                    combined_sign = x_sign_f * w_sign_f[j]
                    values = combined_sign * lut[combined_exp]
                    result[:, j] = values.sum(dim=1)
            
            return result
        
        start = time.perf_counter()
        result_phi = phi_matmul_pytorch_vec(x_exp_t, x_sign_t, w_exp_t, w_sign_t, lut_t)
        torch.cuda.synchronize()
        phi_time = (time.perf_counter() - start) * 1000
        
        print(f"φ-integer (PyTorch vec): {phi_time:.2f} ms")
        
        result_float_np = result_float.cpu().numpy()
        result_phi_np = result_phi.cpu().numpy()
        
        corr = np.corrcoef(result_float_np.flatten(), result_phi_np.flatten())[0, 1]
        print(f"Correlation: {corr:.6f}")


if __name__ == "__main__":
    benchmark()
