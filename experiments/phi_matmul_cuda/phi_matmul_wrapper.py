"""
PyTorch wrapper for φ-Level Fused Matmul CUDA Kernel

This module provides a Python interface to the custom CUDA kernel
that implements fused dequantization + matrix multiplication for
φ-level encoded weights.

Usage:
    from phi_matmul_wrapper import PhiMatmul, encode_phi_weights
    
    # Encode weights
    signs, levels = encode_phi_weights(W_float)
    
    # Create module
    phi_mm = PhiMatmul(signs, levels)
    
    # Forward pass
    output = phi_mm(input)
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import numpy as np
import os

PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)
PHI_LUT_SIZE = 128
PHI_LUT_OFFSET = 64

# CUDA source code (inline compilation)
CUDA_SOURCE = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define PHI 1.6180339887498949f
#define PHI_LUT_SIZE 128
#define PHI_LUT_OFFSET 64

// Optimized kernel with better memory access patterns
// Uses larger tiles and caches signs/levels in shared memory
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32
#define THREAD_M 4
#define THREAD_N 4

__global__ void phi_matmul_optimized_kernel(
    const float* __restrict__ input,
    const int8_t* __restrict__ signs,
    const int8_t* __restrict__ levels,
    float* __restrict__ output,
    const float* __restrict__ phi_lut,
    int M, int N, int K
) {
    // Shared memory for tiles
    __shared__ float s_input[BLOCK_M][BLOCK_K];
    __shared__ int8_t s_signs[BLOCK_N][BLOCK_K];
    __shared__ int8_t s_levels[BLOCK_N][BLOCK_K];
    __shared__ float s_phi_lut[PHI_LUT_SIZE];
    
    // Thread indices
    int tx = threadIdx.x;  // 0-15
    int ty = threadIdx.y;  // 0-15
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;  // 256
    
    // Block indices
    int bm = blockIdx.y * BLOCK_M;
    int bn = blockIdx.x * BLOCK_N;
    
    // Load φ-LUT into shared memory (once per block)
    for (int i = tid; i < PHI_LUT_SIZE; i += num_threads) {
        s_phi_lut[i] = phi_lut[i];
    }
    __syncthreads();
    
    // Each thread computes a THREAD_M x THREAD_N tile of output
    float acc[THREAD_M][THREAD_N] = {{0.0f}};
    
    // Loop over K dimension in blocks
    for (int bk = 0; bk < K; bk += BLOCK_K) {
        // Cooperative load of input tile [BLOCK_M x BLOCK_K]
        // Each thread loads multiple elements
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += num_threads) {
            int load_m = i / BLOCK_K;
            int load_k = i % BLOCK_K;
            int global_m = bm + load_m;
            int global_k = bk + load_k;
            
            if (global_m < M && global_k < K) {
                s_input[load_m][load_k] = input[global_m * K + global_k];
            } else {
                s_input[load_m][load_k] = 0.0f;
            }
        }
        
        // Cooperative load of signs/levels tile [BLOCK_N x BLOCK_K]
        for (int i = tid; i < BLOCK_N * BLOCK_K; i += num_threads) {
            int load_n = i / BLOCK_K;
            int load_k = i % BLOCK_K;
            int global_n = bn + load_n;
            int global_k = bk + load_k;
            
            if (global_n < N && global_k < K) {
                s_signs[load_n][load_k] = signs[global_n * K + global_k];
                s_levels[load_n][load_k] = levels[global_n * K + global_k];
            } else {
                s_signs[load_n][load_k] = 0;
                s_levels[load_n][load_k] = 0;
            }
        }
        __syncthreads();
        
        // Compute partial products
        // Each thread handles THREAD_M rows and THREAD_N columns
        int thread_m_base = ty * THREAD_M;
        int thread_n_base = tx * THREAD_N;
        
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            // Load input values for this thread's rows
            float input_vals[THREAD_M];
            #pragma unroll
            for (int tm = 0; tm < THREAD_M; tm++) {
                input_vals[tm] = s_input[thread_m_base + tm][k];
            }
            
            // Load and dequantize weights for this thread's columns
            float weight_vals[THREAD_N];
            #pragma unroll
            for (int tn = 0; tn < THREAD_N; tn++) {
                int8_t sign = s_signs[thread_n_base + tn][k];
                int8_t level = s_levels[thread_n_base + tn][k];
                weight_vals[tn] = (float)sign * s_phi_lut[level + PHI_LUT_OFFSET];
            }
            
            // Accumulate outer product
            #pragma unroll
            for (int tm = 0; tm < THREAD_M; tm++) {
                #pragma unroll
                for (int tn = 0; tn < THREAD_N; tn++) {
                    acc[tm][tn] += input_vals[tm] * weight_vals[tn];
                }
            }
        }
        __syncthreads();
    }
    
    // Write output
    int thread_m_base = ty * THREAD_M;
    int thread_n_base = tx * THREAD_N;
    
    #pragma unroll
    for (int tm = 0; tm < THREAD_M; tm++) {
        #pragma unroll
        for (int tn = 0; tn < THREAD_N; tn++) {
            int global_m = bm + thread_m_base + tm;
            int global_n = bn + thread_n_base + tn;
            if (global_m < M && global_n < N) {
                output[global_m * N + global_n] = acc[tm][tn];
            }
        }
    }
}

torch::Tensor phi_matmul_cuda(
    torch::Tensor input,
    torch::Tensor signs,
    torch::Tensor levels,
    torch::Tensor phi_lut
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(signs.is_cuda(), "signs must be CUDA tensor");
    TORCH_CHECK(levels.is_cuda(), "levels must be CUDA tensor");
    TORCH_CHECK(phi_lut.is_cuda(), "phi_lut must be CUDA tensor");
    
    int M = input.size(0);
    int K = input.size(1);
    int N = signs.size(0);
    
    TORCH_CHECK(signs.size(1) == K, "signs K dimension mismatch");
    TORCH_CHECK(levels.size(0) == N && levels.size(1) == K, "levels dimension mismatch");
    
    auto output = torch::zeros({M, N}, input.options());
    
    // 16x16 threads, each computing 4x4 output = 64x64 block
    dim3 block(16, 16);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    
    phi_matmul_optimized_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        signs.data_ptr<int8_t>(),
        levels.data_ptr<int8_t>(),
        output.data_ptr<float>(),
        phi_lut.data_ptr<float>(),
        M, N, K
    );
    
    return output;
}
"""

CPP_SOURCE = """
torch::Tensor phi_matmul_cuda(
    torch::Tensor input,
    torch::Tensor signs,
    torch::Tensor levels,
    torch::Tensor phi_lut
);
"""

# Lazy compilation
_phi_matmul_module = None

def get_phi_matmul_module():
    """Compile and cache the CUDA module."""
    global _phi_matmul_module
    if _phi_matmul_module is None:
        _phi_matmul_module = load_inline(
            name='phi_matmul_cuda',
            cpp_sources=[CPP_SOURCE],
            cuda_sources=[CUDA_SOURCE],
            functions=['phi_matmul_cuda'],
            verbose=True,
            extra_cuda_cflags=['-O3', '--use_fast_math']
        )
    return _phi_matmul_module


def create_phi_lut(device='cuda'):
    """Create the φ-level lookup table."""
    lut = torch.tensor(
        [PHI ** (i - PHI_LUT_OFFSET) for i in range(PHI_LUT_SIZE)],
        dtype=torch.float32,
        device=device
    )
    return lut


def encode_phi_weights(W):
    """
    Encode float weights into φ-level format.
    
    Args:
        W: Float tensor of shape (N, K)
        
    Returns:
        signs: Int8 tensor of shape (N, K), values in {-1, +1}
        levels: Int8 tensor of shape (N, K), level indices
    """
    signs = torch.sign(W)
    signs[signs == 0] = 1
    signs = signs.to(torch.int8)
    
    levels = torch.round(torch.log(torch.abs(W) + 1e-45) / LOG_PHI)
    levels = levels.clamp(-PHI_LUT_OFFSET, PHI_LUT_OFFSET - 1).to(torch.int8)
    
    return signs, levels


def decode_phi_weights(signs, levels, phi_lut=None):
    """
    Decode φ-level weights back to float.
    
    Args:
        signs: Int8 tensor of shape (N, K)
        levels: Int8 tensor of shape (N, K)
        phi_lut: Optional precomputed LUT
        
    Returns:
        W: Float tensor of shape (N, K)
    """
    if phi_lut is None:
        phi_lut = create_phi_lut(signs.device)
    
    W = signs.float() * phi_lut[levels.long() + PHI_LUT_OFFSET]
    return W


class PhiMatmul(nn.Module):
    """
    φ-Level Fused Matmul Module
    
    Stores weights in φ-encoded format (sign + level) and performs
    fused dequantization + matmul using a custom CUDA kernel.
    
    Args:
        signs: Int8 tensor of shape (out_features, in_features)
        levels: Int8 tensor of shape (out_features, in_features)
    """
    
    def __init__(self, signs, levels):
        super().__init__()
        self.register_buffer('signs', signs.contiguous())
        self.register_buffer('levels', levels.contiguous())
        self.register_buffer('phi_lut', create_phi_lut(signs.device))
        
        self.out_features = signs.shape[0]
        self.in_features = signs.shape[1]
        
        self._cuda_module = None
    
    @classmethod
    def from_float(cls, W):
        """Create from float weight matrix."""
        signs, levels = encode_phi_weights(W)
        return cls(signs.cuda(), levels.cuda())
    
    def forward(self, x):
        """
        Forward pass using fused CUDA kernel.
        
        Args:
            x: Input tensor of shape (batch, in_features) or (in_features,)
            
        Returns:
            Output tensor of shape (batch, out_features) or (out_features,)
        """
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        
        # Ensure contiguous float32
        x = x.contiguous().float()
        
        # Try CUDA kernel, fall back to PyTorch if compilation fails
        try:
            if self._cuda_module is None:
                self._cuda_module = get_phi_matmul_module()
            
            output = self._cuda_module.phi_matmul_cuda(
                x, self.signs, self.levels, self.phi_lut
            )
        except Exception as e:
            # Fallback to PyTorch implementation
            print(f"CUDA kernel failed ({e}), using PyTorch fallback")
            W = decode_phi_weights(self.signs, self.levels, self.phi_lut)
            output = x @ W.T
        
        if squeeze:
            output = output.squeeze(0)
        
        return output
    
    def to_float(self):
        """Decode weights to float tensor."""
        return decode_phi_weights(self.signs, self.levels, self.phi_lut)


class PhiLinear(nn.Module):
    """
    φ-Level Linear Layer (drop-in replacement for nn.Linear)
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to include bias (default: True)
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize with random φ-level weights
        W = torch.randn(out_features, in_features) * 0.02
        signs, levels = encode_phi_weights(W)
        
        self.register_buffer('signs', signs)
        self.register_buffer('levels', levels)
        self.register_buffer('phi_lut', create_phi_lut('cpu'))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._cuda_module = None
    
    @classmethod
    def from_linear(cls, linear):
        """Convert nn.Linear to PhiLinear."""
        phi_linear = cls(linear.in_features, linear.out_features, 
                         bias=linear.bias is not None)
        
        signs, levels = encode_phi_weights(linear.weight.data)
        phi_linear.signs = signs.to(linear.weight.device)
        phi_linear.levels = levels.to(linear.weight.device)
        phi_linear.phi_lut = create_phi_lut(linear.weight.device)
        
        if linear.bias is not None:
            phi_linear.bias.data = linear.bias.data.clone()
        
        return phi_linear
    
    def forward(self, x):
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        
        x = x.contiguous().float()
        
        try:
            if self._cuda_module is None:
                self._cuda_module = get_phi_matmul_module()
            
            output = self._cuda_module.phi_matmul_cuda(
                x, self.signs, self.levels, self.phi_lut
            )
        except Exception:
            W = decode_phi_weights(self.signs, self.levels, self.phi_lut)
            output = x @ W.T
        
        if self.bias is not None:
            output = output + self.bias
        
        if squeeze:
            output = output.squeeze(0)
        
        return output


def benchmark_phi_matmul(M=1, N=4096, K=4096, n_iter=1000, warmup=100):
    """
    Benchmark φ-matmul vs standard matmul.
    
    Args:
        M: Batch size
        N: Output features
        K: Input features
        n_iter: Number of iterations
        warmup: Warmup iterations
    """
    import time
    
    print(f"Benchmarking: M={M}, N={N}, K={K}")
    
    # Create test data
    x = torch.randn(M, K, device='cuda', dtype=torch.float32)
    W = torch.randn(N, K, device='cuda', dtype=torch.float32) * 0.02
    
    # Encode to φ-level
    signs, levels = encode_phi_weights(W)
    signs = signs.cuda()
    levels = levels.cuda()
    phi_lut = create_phi_lut('cuda')
    
    # Decode for comparison
    W_phi = decode_phi_weights(signs, levels, phi_lut)
    
    # Verify correctness
    y_ref = x @ W.T
    y_phi_ref = x @ W_phi.T
    
    corr = torch.corrcoef(torch.stack([y_ref.flatten(), y_phi_ref.flatten()]))[0, 1]
    print(f"φ-encoding correlation: {corr.item()*100:.4f}%")
    
    # Try CUDA kernel
    try:
        module = get_phi_matmul_module()
        
        # Warmup
        for _ in range(warmup):
            _ = module.phi_matmul_cuda(x, signs, levels, phi_lut)
        torch.cuda.synchronize()
        
        # Benchmark CUDA kernel
        start = time.perf_counter()
        for _ in range(n_iter):
            _ = module.phi_matmul_cuda(x, signs, levels, phi_lut)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) / n_iter * 1000
        
        # Verify CUDA output
        y_cuda = module.phi_matmul_cuda(x, signs, levels, phi_lut)
        cuda_corr = torch.corrcoef(torch.stack([y_phi_ref.flatten(), y_cuda.flatten()]))[0, 1]
        print(f"CUDA kernel correlation: {cuda_corr.item()*100:.4f}%")
        print(f"CUDA kernel time: {cuda_time:.4f} ms")
        
    except Exception as e:
        print(f"CUDA kernel compilation failed: {e}")
        cuda_time = None
    
    # Benchmark standard matmul
    for _ in range(warmup):
        _ = x @ W.T
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = x @ W.T
    torch.cuda.synchronize()
    std_time = (time.perf_counter() - start) / n_iter * 1000
    
    print(f"Standard matmul time: {std_time:.4f} ms")
    
    if cuda_time:
        print(f"Speedup: {std_time / cuda_time:.2f}x")
    
    return cuda_time, std_time


if __name__ == '__main__':
    print("=" * 60)
    print("φ-Level Fused Matmul CUDA Kernel Benchmark")
    print("=" * 60)
    
    # Test various sizes
    for M in [1, 4, 16]:
        for N in [4096, 8192]:
            for K in [4096, 8192]:
                print()
                benchmark_phi_matmul(M=M, N=N, K=K, n_iter=500, warmup=50)
