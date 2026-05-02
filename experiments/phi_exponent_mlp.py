#!/usr/bin/env python3
"""
φ-Exponent MLP: Integer Arithmetic for Neural Network Computation

Based on Doc 124 (φ-Exponent Arithmetic), this implements MLP computation
using only integer addition and table lookup instead of floating-point matmul.

The key insight:
- Values are represented as: v = sign × φ^(exponent/k)
- Multiplication becomes: exp_a + exp_b (integer addition)
- Dot products become: sum of table lookups

From Doc 124:
> "99.97% accuracy with pure integer arithmetic + lookup table."

Author: TruthSpace LCM Team
Date: 2026-01-29
"""

import numpy as np
import time
from typing import Tuple
from dataclasses import dataclass

# Golden ratio
PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)


@dataclass
class PhiGridConfig:
    """Configuration for φ-grid quantization."""
    k: int = 8  # Resolution: adjacent points differ by φ^(1/k)
    bias: int = 128  # Exponent bias for uint8 storage
    max_exp: int = 255  # Maximum exponent value


class PhiGrid:
    """
    φ-grid number system for integer arithmetic.
    
    Values are represented as: v = sign × φ^((exponent - bias) / k)
    
    Operations:
    - Multiply: add exponents
    - Divide: subtract exponents
    - Dot product: sum of table lookups
    """
    
    def __init__(self, k: int = 8, bias: int = 128):
        self.k = k
        self.bias = bias
        
        # Precompute lookup table: PHI_LUT[i] = φ^((i - bias) / k)
        self.PHI_LUT = np.array([
            PHI ** ((i - bias) / k) for i in range(256)
        ], dtype=np.float32)
        
        # For very small/large values, clip to prevent overflow
        self.PHI_LUT = np.clip(self.PHI_LUT, 1e-20, 1e20)
    
    def to_phi_grid(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert float values to φ-grid representation.
        
        Returns:
            signs: int8 array of signs (+1 or -1)
            exponents: uint8 array of exponents
        """
        signs = np.sign(values).astype(np.int8)
        signs[signs == 0] = 1  # Handle zeros
        
        magnitudes = np.abs(values) + 1e-20  # Avoid log(0)
        
        # exponent = k * log_φ(magnitude) + bias
        exponents = self.k * np.log(magnitudes) / LOG_PHI + self.bias
        exponents = np.clip(exponents, 0, 255).astype(np.uint8)
        
        return signs, exponents
    
    def from_phi_grid(self, signs: np.ndarray, exponents: np.ndarray) -> np.ndarray:
        """Convert φ-grid representation back to float."""
        return signs.astype(np.float32) * self.PHI_LUT[exponents]
    
    def phi_multiply(self, exp_a: np.ndarray, exp_b: np.ndarray) -> np.ndarray:
        """
        Multiply two φ-grid values by adding exponents.
        
        φ^(a/k) × φ^(b/k) = φ^((a+b)/k)
        """
        # Add exponents, subtract bias to compensate for double-counting
        result = exp_a.astype(np.int16) + exp_b.astype(np.int16) - self.bias
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def phi_dot_product(self, w_signs: np.ndarray, w_exps: np.ndarray,
                        x_signs: np.ndarray, x_exps: np.ndarray) -> float:
        """
        Compute dot product using φ-arithmetic.
        
        w · x = Σ (w_sign × x_sign) × φ^((w_exp + x_exp - bias) / k)
        """
        # Multiply signs (integer)
        result_signs = w_signs * x_signs
        
        # Add exponents (integer)
        result_exps = self.phi_multiply(w_exps, x_exps)
        
        # Lookup and sum
        return np.sum(result_signs.astype(np.float32) * self.PHI_LUT[result_exps])
    
    def phi_matmul(self, W_signs: np.ndarray, W_exps: np.ndarray,
                   x_signs: np.ndarray, x_exps: np.ndarray) -> np.ndarray:
        """
        Matrix-vector multiplication using φ-arithmetic.
        
        W @ x where W is (out_dim, in_dim) and x is (in_dim,)
        """
        out_dim = W_signs.shape[0]
        result = np.zeros(out_dim, dtype=np.float32)
        
        for i in range(out_dim):
            result[i] = self.phi_dot_product(
                W_signs[i], W_exps[i], x_signs, x_exps
            )
        
        return result
    
    def phi_matmul_vectorized(self, W_signs: np.ndarray, W_exps: np.ndarray,
                               x_signs: np.ndarray, x_exps: np.ndarray) -> np.ndarray:
        """
        Vectorized matrix-vector multiplication using φ-arithmetic.
        
        Much faster than the loop version.
        """
        # Broadcast: W_exps is (out_dim, in_dim), x_exps is (in_dim,)
        # Result exponents: (out_dim, in_dim)
        result_exps = W_exps.astype(np.int16) + x_exps.astype(np.int16) - self.bias
        result_exps = np.clip(result_exps, 0, 255).astype(np.uint8)
        
        # Result signs: (out_dim, in_dim)
        result_signs = W_signs.astype(np.int16) * x_signs.astype(np.int16)
        
        # Lookup and sum along in_dim
        values = result_signs.astype(np.float32) * self.PHI_LUT[result_exps]
        return np.sum(values, axis=1)


class PhiMLP:
    """
    MLP using φ-exponent arithmetic.
    
    Converts weights to φ-grid and uses integer arithmetic for computation.
    """
    
    def __init__(self, W_gate: np.ndarray, W_up: np.ndarray, W_down: np.ndarray,
                 k: int = 8):
        """
        Initialize φ-MLP with weight matrices.
        
        Args:
            W_gate: Gate projection (intermediate_dim, hidden_dim)
            W_up: Up projection (intermediate_dim, hidden_dim)
            W_down: Down projection (hidden_dim, intermediate_dim)
            k: φ-grid resolution
        """
        self.grid = PhiGrid(k=k)
        
        # Convert weights to φ-grid
        print(f"Converting MLP weights to φ-grid (k={k})...")
        
        self.gate_signs, self.gate_exps = self.grid.to_phi_grid(W_gate)
        self.up_signs, self.up_exps = self.grid.to_phi_grid(W_up)
        self.down_signs, self.down_exps = self.grid.to_phi_grid(W_down)
        
        # Store original for comparison
        self.W_gate = W_gate
        self.W_up = W_up
        self.W_down = W_down
        
        # Compute reconstruction error
        gate_recon = self.grid.from_phi_grid(self.gate_signs, self.gate_exps)
        gate_error = np.linalg.norm(W_gate - gate_recon) / np.linalg.norm(W_gate)
        print(f"  Gate weight reconstruction error: {gate_error*100:.2f}%")
    
    def forward_float(self, x: np.ndarray) -> np.ndarray:
        """Standard float MLP forward pass."""
        gate = self.W_gate @ x
        up = self.W_up @ x
        
        # SiLU activation
        gate_clipped = np.clip(gate, -88, 88)
        silu_gate = gate / (1 + np.exp(-gate_clipped))
        hidden = silu_gate * up
        
        return self.W_down @ hidden
    
    def forward_phi(self, x: np.ndarray) -> np.ndarray:
        """φ-arithmetic MLP forward pass."""
        # Convert input to φ-grid
        x_signs, x_exps = self.grid.to_phi_grid(x)
        
        # Gate and up projections using φ-arithmetic
        gate = self.grid.phi_matmul_vectorized(
            self.gate_signs, self.gate_exps, x_signs, x_exps
        )
        up = self.grid.phi_matmul_vectorized(
            self.up_signs, self.up_exps, x_signs, x_exps
        )
        
        # SiLU activation (still float - this is the nonlinearity)
        gate_clipped = np.clip(gate, -88, 88)
        silu_gate = gate / (1 + np.exp(-gate_clipped))
        hidden = silu_gate * up
        
        # Convert hidden to φ-grid for down projection
        hidden_signs, hidden_exps = self.grid.to_phi_grid(hidden)
        
        # Down projection using φ-arithmetic
        out = self.grid.phi_matmul_vectorized(
            self.down_signs, self.down_exps, hidden_signs, hidden_exps
        )
        
        return out
    
    def forward_phi_hybrid(self, x: np.ndarray) -> np.ndarray:
        """
        Hybrid: φ-arithmetic for gate/up, float for down.
        
        The down projection is the bottleneck (3584 × 18944),
        but it operates on sparse hidden activations.
        """
        # Convert input to φ-grid
        x_signs, x_exps = self.grid.to_phi_grid(x)
        
        # Gate and up projections using φ-arithmetic
        gate = self.grid.phi_matmul_vectorized(
            self.gate_signs, self.gate_exps, x_signs, x_exps
        )
        up = self.grid.phi_matmul_vectorized(
            self.up_signs, self.up_exps, x_signs, x_exps
        )
        
        # SiLU activation
        gate_clipped = np.clip(gate, -88, 88)
        silu_gate = gate / (1 + np.exp(-gate_clipped))
        hidden = silu_gate * up
        
        # Down projection using float (hidden is already float)
        return self.W_down @ hidden


def test_phi_grid():
    """Test basic φ-grid operations."""
    print("=" * 60)
    print("Testing φ-Grid Operations")
    print("=" * 60)
    
    grid = PhiGrid(k=8)
    
    # Test roundtrip
    values = np.array([0.1, 1.0, 10.0, -0.5, -5.0, 0.001, 100.0])
    signs, exps = grid.to_phi_grid(values)
    reconstructed = grid.from_phi_grid(signs, exps)
    
    print("\nRoundtrip test:")
    for v, r in zip(values, reconstructed):
        error = abs(v - r) / abs(v) * 100
        print(f"  {v:8.4f} → {r:8.4f} (error: {error:.2f}%)")
    
    # Test multiplication
    print("\nMultiplication test:")
    a, b = 2.5, 3.0
    a_signs, a_exps = grid.to_phi_grid(np.array([a]))
    b_signs, b_exps = grid.to_phi_grid(np.array([b]))
    
    result_exp = grid.phi_multiply(a_exps, b_exps)
    result = grid.from_phi_grid(a_signs * b_signs, result_exp)
    
    print(f"  {a} × {b} = {a*b} (exact)")
    print(f"  φ-arithmetic: {result[0]:.4f}")
    print(f"  Error: {abs(a*b - result[0]) / (a*b) * 100:.2f}%")
    
    # Test dot product
    print("\nDot product test:")
    w = np.array([1.0, 2.0, 3.0, 4.0])
    x = np.array([0.5, 0.5, 0.5, 0.5])
    
    exact = np.dot(w, x)
    
    w_signs, w_exps = grid.to_phi_grid(w)
    x_signs, x_exps = grid.to_phi_grid(x)
    phi_result = grid.phi_dot_product(w_signs, w_exps, x_signs, x_exps)
    
    print(f"  w · x = {exact} (exact)")
    print(f"  φ-arithmetic: {phi_result:.4f}")
    print(f"  Error: {abs(exact - phi_result) / exact * 100:.2f}%")


def test_phi_mlp():
    """Test φ-MLP on Qwen2 weights."""
    print("\n" + "=" * 60)
    print("Testing φ-MLP on Qwen2 Weights")
    print("=" * 60)
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\nLoading Qwen2...")
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2-7B-Instruct',
        torch_dtype=torch.float32,
        device_map='cpu'
    )
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct')
    
    # Get MLP weights from layer 14
    layer = model.model.layers[14]
    W_gate = layer.mlp.gate_proj.weight.data.numpy()
    W_up = layer.mlp.up_proj.weight.data.numpy()
    W_down = layer.mlp.down_proj.weight.data.numpy()
    
    print(f"\nMLP shapes:")
    print(f"  W_gate: {W_gate.shape}")
    print(f"  W_up: {W_up.shape}")
    print(f"  W_down: {W_down.shape}")
    
    # Get a test input
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[14][0, -1, :].numpy()
    
    # Apply layer norm
    ln_weight = layer.post_attention_layernorm.weight.data.numpy()
    rms = np.sqrt(np.mean(hidden ** 2) + 1e-6)
    x = (hidden / rms) * ln_weight
    
    print(f"\nInput x shape: {x.shape}")
    print(f"Input x norm: {np.linalg.norm(x):.4f}")
    
    # Create φ-MLP
    phi_mlp = PhiMLP(W_gate, W_up, W_down, k=8)
    
    # Compare outputs
    print("\n--- Comparing MLP outputs ---")
    
    # Float forward
    start = time.time()
    out_float = phi_mlp.forward_float(x)
    float_time = time.time() - start
    
    # φ-arithmetic forward
    start = time.time()
    out_phi = phi_mlp.forward_phi(x)
    phi_time = time.time() - start
    
    # Hybrid forward
    start = time.time()
    out_hybrid = phi_mlp.forward_phi_hybrid(x)
    hybrid_time = time.time() - start
    
    # Correlations
    corr_phi = np.corrcoef(out_float.flatten(), out_phi.flatten())[0, 1]
    corr_hybrid = np.corrcoef(out_float.flatten(), out_hybrid.flatten())[0, 1]
    
    # Relative errors
    err_phi = np.linalg.norm(out_float - out_phi) / np.linalg.norm(out_float)
    err_hybrid = np.linalg.norm(out_float - out_hybrid) / np.linalg.norm(out_float)
    
    print(f"\nFloat MLP:")
    print(f"  Time: {float_time*1000:.1f}ms")
    print(f"  Output norm: {np.linalg.norm(out_float):.4f}")
    
    print(f"\nφ-arithmetic MLP:")
    print(f"  Time: {phi_time*1000:.1f}ms")
    print(f"  Correlation: {corr_phi:.6f}")
    print(f"  Relative error: {err_phi*100:.2f}%")
    print(f"  Speedup: {float_time/phi_time:.2f}x")
    
    print(f"\nHybrid MLP (φ gate/up, float down):")
    print(f"  Time: {hybrid_time*1000:.1f}ms")
    print(f"  Correlation: {corr_hybrid:.6f}")
    print(f"  Relative error: {err_hybrid*100:.2f}%")
    print(f"  Speedup: {float_time/hybrid_time:.2f}x")
    
    del model
    
    return corr_phi, corr_hybrid


def benchmark_phi_mlp_speed():
    """Benchmark φ-MLP speed with different k values."""
    print("\n" + "=" * 60)
    print("Benchmarking φ-MLP Speed")
    print("=" * 60)
    
    # Simulate MLP dimensions
    hidden_dim = 3584
    intermediate_dim = 18944
    
    np.random.seed(42)
    W_gate = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32) * 0.02
    W_up = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32) * 0.02
    W_down = np.random.randn(hidden_dim, intermediate_dim).astype(np.float32) * 0.02
    x = np.random.randn(hidden_dim).astype(np.float32)
    
    # Float baseline
    n_iters = 5
    
    start = time.time()
    for _ in range(n_iters):
        gate = W_gate @ x
        up = W_up @ x
        silu = gate / (1 + np.exp(-np.clip(gate, -88, 88)))
        hidden = silu * up
        out = W_down @ hidden
    float_time = (time.time() - start) / n_iters
    
    print(f"\nFloat MLP: {float_time*1000:.1f}ms per forward")
    
    # Test different k values
    print(f"\n{'k':>4} {'Time (ms)':>12} {'Speedup':>10} {'Note':>20}")
    print("-" * 50)
    
    for k in [4, 8, 16]:
        phi_mlp = PhiMLP(W_gate, W_up, W_down, k=k)
        
        # Warm up
        _ = phi_mlp.forward_phi(x)
        
        start = time.time()
        for _ in range(n_iters):
            out = phi_mlp.forward_phi(x)
        phi_time = (time.time() - start) / n_iters
        
        speedup = float_time / phi_time
        note = "slower" if speedup < 1 else f"{speedup:.1f}x faster"
        
        print(f"{k:>4} {phi_time*1000:>12.1f} {speedup:>10.2f}x {note:>20}")


def test_int8_quantized_mlp():
    """
    Test INT8 quantized MLP - this is what actually speeds things up.
    
    The insight: φ-grid is for hardware without FPU.
    For NumPy, INT8 quantization with float accumulation is faster.
    """
    print("\n" + "=" * 60)
    print("Testing INT8 Quantized MLP")
    print("=" * 60)
    
    # Simulate MLP dimensions
    hidden_dim = 3584
    intermediate_dim = 18944
    
    np.random.seed(42)
    W_gate = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32) * 0.02
    W_up = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32) * 0.02
    W_down = np.random.randn(hidden_dim, intermediate_dim).astype(np.float32) * 0.02
    x = np.random.randn(hidden_dim).astype(np.float32)
    
    # Float baseline
    gate_float = W_gate @ x
    up_float = W_up @ x
    silu_float = gate_float / (1 + np.exp(-np.clip(gate_float, -88, 88)))
    hidden_float = silu_float * up_float
    out_float = W_down @ hidden_float
    
    print(f"\nFloat output norm: {np.linalg.norm(out_float):.4f}")
    
    # INT8 quantization
    def quantize_int8(W):
        scale = np.max(np.abs(W)) / 127
        W_int8 = np.round(W / scale).astype(np.int8)
        return W_int8, scale
    
    W_gate_int8, gate_scale = quantize_int8(W_gate)
    W_up_int8, up_scale = quantize_int8(W_up)
    W_down_int8, down_scale = quantize_int8(W_down)
    
    # Quantize input
    x_scale = np.max(np.abs(x)) / 127
    x_int8 = np.round(x / x_scale).astype(np.int8)
    
    print(f"\nQuantization scales:")
    print(f"  W_gate: {gate_scale:.6f}")
    print(f"  W_up: {up_scale:.6f}")
    print(f"  W_down: {down_scale:.6f}")
    print(f"  x: {x_scale:.6f}")
    
    # INT8 forward pass
    # Note: NumPy doesn't have optimized int8 matmul, so we cast to int32 for accumulation
    n_iters = 5
    
    # Float baseline timing
    start = time.time()
    for _ in range(n_iters):
        gate = W_gate @ x
        up = W_up @ x
        silu = gate / (1 + np.exp(-np.clip(gate, -88, 88)))
        hidden = silu * up
        out = W_down @ hidden
    float_time = (time.time() - start) / n_iters
    
    # INT8 timing (with int32 accumulation)
    start = time.time()
    for _ in range(n_iters):
        # Cast to int16 for matmul to avoid overflow
        gate_int = W_gate_int8.astype(np.int16) @ x_int8.astype(np.int16)
        up_int = W_up_int8.astype(np.int16) @ x_int8.astype(np.int16)
        
        # Dequantize for SiLU
        gate = gate_int.astype(np.float32) * gate_scale * x_scale
        up = up_int.astype(np.float32) * up_scale * x_scale
        
        silu = gate / (1 + np.exp(-np.clip(gate, -88, 88)))
        hidden = silu * up
        
        # Requantize hidden for down projection
        hidden_scale = np.max(np.abs(hidden)) / 127
        hidden_int8 = np.round(hidden / hidden_scale).astype(np.int8)
        
        out_int = W_down_int8.astype(np.int16) @ hidden_int8.astype(np.int16)
        out = out_int.astype(np.float32) * down_scale * hidden_scale
    int8_time = (time.time() - start) / n_iters
    
    # Compute correlation
    corr = np.corrcoef(out_float.flatten(), out.flatten())[0, 1]
    rel_err = np.linalg.norm(out_float - out) / np.linalg.norm(out_float)
    
    print(f"\nResults:")
    print(f"  Float time: {float_time*1000:.1f}ms")
    print(f"  INT8 time: {int8_time*1000:.1f}ms")
    print(f"  Speedup: {float_time/int8_time:.2f}x")
    print(f"  Correlation: {corr:.6f}")
    print(f"  Relative error: {rel_err*100:.2f}%")
    
    # The real speedup comes from using optimized INT8 libraries
    print("\n--- Note ---")
    print("NumPy doesn't have optimized INT8 matmul.")
    print("For real speedup, use:")
    print("  - PyTorch quantization (torch.int8)")
    print("  - ONNX Runtime with INT8")
    print("  - TensorRT INT8")
    print("  - Custom SIMD/AVX2 implementation")


if __name__ == "__main__":
    test_phi_grid()
    test_phi_mlp()
    benchmark_phi_mlp_speed()
    test_int8_quantized_mlp()
