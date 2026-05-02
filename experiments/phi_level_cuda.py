#!/usr/bin/env python3
"""
φ-Level CUDA Kernel - Direct Compressed Matmul

Key insight: Read 1 byte per weight (7 bits used), decompress in registers.
This halves memory bandwidth vs bfloat16.

Format: 1 byte per weight
  - Bits 0-5: level offset (0-63, maps to φ^(level_min + offset))
  - Bit 6: sign (0=positive, 1=negative)
  - Bit 7: unused

Decompression: value = (1 - 2*sign) * φ^(level_min + offset)
This is just one LUT lookup and one multiply.
"""

import torch
import numpy as np
import time
from typing import Tuple, Optional
import cupy as cp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)

# CUDA kernel for φ-level matmul - optimized with tiling
PHI_MATMUL_KERNEL = r"""
extern "C" {

#define TILE_SIZE 128
#define BLOCK_SIZE 256

__global__ void phi_level_matmul(
    const unsigned char* __restrict__ W_packed,  // (out_dim, in_dim) packed weights
    const float* __restrict__ x,                  // (batch, in_dim) input
    const float* __restrict__ phi_lut,            // (64,) precomputed φ^level values
    float* __restrict__ y,                        // (batch, out_dim) output
    int batch,
    int out_dim,
    int in_dim
) {
    // Load LUT into shared memory
    __shared__ float s_phi_lut[64];
    __shared__ float s_x[TILE_SIZE];
    
    if (threadIdx.x < 64) {
        s_phi_lut[threadIdx.x] = phi_lut[threadIdx.x];
    }
    __syncthreads();
    
    // Each thread computes one output element
    int b = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= batch || j >= out_dim) return;
    
    const unsigned char* W_row = W_packed + (long long)j * in_dim;
    const float* x_row = x + (long long)b * in_dim;
    
    float sum = 0.0f;
    
    // Process in tiles for better cache utilization
    for (int tile_start = 0; tile_start < in_dim; tile_start += TILE_SIZE) {
        // Cooperatively load x tile into shared memory
        int tile_end = min(tile_start + TILE_SIZE, in_dim);
        int tile_len = tile_end - tile_start;
        
        // Each thread loads some elements
        for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
            s_x[i] = x_row[tile_start + i];
        }
        __syncthreads();
        
        // Process this tile
        for (int i = 0; i < tile_len; i++) {
            unsigned char w = W_row[tile_start + i];
            float x_val = s_x[i];
            
            int level = w & 0x3F;
            float sign = (w & 0x40) ? -1.0f : 1.0f;
            float phi_val = s_phi_lut[level];
            
            sum += sign * phi_val * x_val;
        }
        __syncthreads();
    }
    
    y[(long long)b * out_dim + j] = sum;
}

}  // extern "C"
"""


class PhiLevelLinearCUDA:
    """
    Linear layer with φ-level compressed weights and custom CUDA kernel.
    
    Reads 1 byte per weight (vs 2 bytes for bfloat16) = 2× bandwidth reduction.
    Decompression happens in registers, not memory.
    """
    
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """
        Initialize from a standard weight tensor.
        
        Args:
            weight: (out_dim, in_dim) weight matrix
            bias: Optional (out_dim,) bias vector
        """
        self.out_dim, self.in_dim = weight.shape
        self.bias = bias
        
        # Compress weights
        self.packed, self.level_min = self._compress(weight)
        
        # Compile CUDA kernel
        self._compile_kernel()
        
        # Transfer to GPU
        self.d_packed = cp.asarray(self.packed)
        if bias is not None:
            self.d_bias = cp.asarray(bias.float().cpu().numpy())
        else:
            self.d_bias = None
    
    def _compress(self, weight: torch.Tensor) -> Tuple[np.ndarray, int]:
        """Compress weight to φ-level packed format."""
        W = weight.float().cpu().numpy()
        
        # Get signs (bit 6)
        signs = (W < 0).astype(np.uint8) << 6
        
        # Get φ-levels
        abs_W = np.maximum(np.abs(W), 1e-45)
        levels = np.round(np.log(abs_W) / LOG_PHI).astype(np.int32)
        
        # Find level range
        level_min = levels.min()
        level_max = levels.max()
        
        # Ensure we fit in 6 bits (0-63)
        if level_max - level_min > 63:
            # Clip to 64 levels centered on the median
            median_level = np.median(levels).astype(np.int32)
            level_min = median_level - 31
            level_max = median_level + 32
            levels = np.clip(levels, level_min, level_max)
        
        # Convert to offset (bits 0-5)
        level_offsets = (levels - level_min).astype(np.uint8)
        
        # Pack: bits 0-5 = level offset, bit 6 = sign
        packed = level_offsets | signs
        
        return packed, level_min
    
    def _compile_kernel(self):
        """Compile the CUDA kernel."""
        self.module = cp.RawModule(code=PHI_MATMUL_KERNEL, options=('-std=c++11',))
        self.matmul_kernel = self.module.get_function('phi_level_matmul')
        
        # Precompute φ^level LUT on CPU, then transfer to GPU
        phi_lut_np = np.array([PHI ** (self.level_min + i) for i in range(64)], dtype=np.float32)
        self.phi_lut = cp.asarray(phi_lut_np)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using φ-level compressed matmul.
        
        Args:
            x: (batch, seq_len, in_dim) or (batch, in_dim) input
        
        Returns:
            (batch, seq_len, out_dim) or (batch, out_dim) output
        """
        # Handle different input shapes
        orig_shape = x.shape
        if x.dim() == 3:
            batch, seq_len, in_dim = x.shape
            x = x.reshape(batch * seq_len, in_dim)
        else:
            batch = x.shape[0]
            seq_len = None
        
        # Convert to cupy
        x_cp = cp.asarray(x.float().cpu().numpy())
        
        # Allocate output
        y_cp = cp.zeros((x_cp.shape[0], self.out_dim), dtype=cp.float32)
        
        # Launch kernel
        block_size = 256
        grid_x = (self.out_dim + block_size - 1) // block_size
        grid_y = x_cp.shape[0]
        
        self.matmul_kernel(
            (grid_x, grid_y), (block_size,),
            (self.d_packed, x_cp, self.phi_lut, y_cp, 
             x_cp.shape[0], self.out_dim, self.in_dim)
        )
        
        # Add bias if present
        if self.d_bias is not None:
            y_cp += self.d_bias
        
        # Convert back to torch
        y = torch.tensor(cp.asnumpy(y_cp), dtype=x.dtype, device=x.device)
        
        # Restore original shape
        if seq_len is not None:
            y = y.reshape(batch, seq_len, self.out_dim)
        
        return y
    
    def get_memory_stats(self) -> dict:
        """Get memory usage statistics."""
        packed_bytes = self.packed.nbytes
        original_bytes = self.out_dim * self.in_dim * 2  # bfloat16
        
        return {
            "packed_bytes": packed_bytes,
            "original_bytes": original_bytes,
            "compression": original_bytes / packed_bytes,
        }


def test_phi_level_cuda():
    """Test the φ-level CUDA kernel."""
    print("=" * 70)
    print("φ-LEVEL CUDA KERNEL TEST")
    print("=" * 70)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\nLoading Qwen2-7B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()
    
    # Test on gate projection
    print("\n--- Testing Gate Projection ---")
    layer = model.model.layers[0]
    W_gate = layer.mlp.gate_proj.weight.data
    
    print(f"Weight shape: {tuple(W_gate.shape)}")
    print(f"Original size: {W_gate.numel() * 2 / 1e6:.2f} MB")
    
    # Create φ-level layer
    start = time.perf_counter()
    phi_layer = PhiLevelLinearCUDA(W_gate)
    init_time = (time.perf_counter() - start) * 1000
    
    stats = phi_layer.get_memory_stats()
    print(f"Compressed size: {stats['packed_bytes'] / 1e6:.2f} MB")
    print(f"Compression: {stats['compression']:.2f}×")
    print(f"Init time: {init_time:.1f} ms")
    
    # Test accuracy
    print("\n--- Accuracy Test ---")
    test_input = torch.randn(1, 5, 3584, dtype=torch.bfloat16, device="cuda")
    
    # Original forward
    with torch.no_grad():
        orig_out = layer.mlp.gate_proj(test_input)
    
    # φ-level forward
    phi_out = phi_layer.forward(test_input)
    
    # Compare
    orig_flat = orig_out.float().flatten().cpu()
    phi_flat = phi_out.float().flatten().cpu()
    
    correlation = torch.corrcoef(torch.stack([orig_flat, phi_flat]))[0, 1].item()
    max_error = (orig_out.float().cpu() - phi_out.float().cpu()).abs().max().item()
    
    print(f"Correlation: {correlation * 100:.4f}%")
    print(f"Max error: {max_error:.6f}")
    
    # Benchmark speed
    print("\n--- Speed Benchmark ---")
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = layer.mlp.gate_proj(test_input)
        _ = phi_layer.forward(test_input)
    torch.cuda.synchronize()
    cp.cuda.Stream.null.synchronize()
    
    # Original
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = layer.mlp.gate_proj(test_input)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    orig_time = np.mean(times)
    
    # φ-level
    times = []
    for _ in range(100):
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        _ = phi_layer.forward(test_input)
        cp.cuda.Stream.null.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    phi_time = np.mean(times)
    
    print(f"Original: {orig_time:.3f} ms")
    print(f"φ-level: {phi_time:.3f} ms")
    print(f"Speedup: {orig_time / phi_time:.2f}×")
    
    # Memory bandwidth analysis
    print("\n--- Bandwidth Analysis ---")
    
    # Original: read W (bf16) + x (bf16), write y (bf16)
    orig_bytes = W_gate.numel() * 2 + test_input.numel() * 2 + (1 * 5 * W_gate.shape[0]) * 2
    orig_bandwidth = orig_bytes / (orig_time / 1000) / 1e9
    
    # φ-level: read W (uint8) + x (float32) + LUT, write y (float32)
    phi_bytes = phi_layer.packed.nbytes + test_input.numel() * 4 + 64 * 4 + (1 * 5 * W_gate.shape[0]) * 4
    phi_bandwidth = phi_bytes / (phi_time / 1000) / 1e9
    
    print(f"Original bandwidth: {orig_bandwidth:.1f} GB/s")
    print(f"φ-level bandwidth: {phi_bandwidth:.1f} GB/s")
    print(f"Theoretical GPU bandwidth: 1008 GB/s")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_phi_level_cuda()
