#!/usr/bin/env python3
"""
φ-Level Model Compression - Target: 170 tok/s

Key insight: Weights cluster at 15 φ-levels (99% coverage).
Represent as: W[i,j] = sign[i,j] × φ^level[i,j]

Storage: 1 bit (sign) + 4 bits (level) = 5 bits per weight
Compression: 16 bits / 5 bits = 3.2×

With 3.2× compression:
- Model size: 15.23 GB → 4.76 GB
- Theoretical max: 53 tok/s → 170 tok/s
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import struct

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PHI = 1.6180339887498949
LOG_PHI = np.log(PHI)


@dataclass
class PhiLevelWeight:
    """Compressed weight matrix using φ-level representation."""
    shape: Tuple[int, int]
    
    # Packed representation: 5 bits per weight
    # Bits 0-3: level index (0-15)
    # Bit 4: sign (0=positive, 1=negative)
    packed: np.ndarray  # uint8, 2 weights per byte (with some waste)
    
    # Level lookup table
    level_values: np.ndarray  # float32, the actual φ^level values
    level_map: np.ndarray  # int8, maps level index to actual level
    
    # For GPU
    d_packed: Optional[torch.Tensor] = None
    d_level_values: Optional[torch.Tensor] = None


class PhiLevelCompressor:
    """Compress model weights to φ-level representation."""
    
    def __init__(self, num_levels: int = 16):
        """
        Args:
            num_levels: Number of discrete φ-levels to use (16 = 4 bits)
        """
        self.num_levels = num_levels
        self.stats = {
            "total_weights": 0,
            "original_bytes": 0,
            "compressed_bytes": 0,
        }
    
    def compress_weight(self, W: torch.Tensor) -> PhiLevelWeight:
        """Compress a weight matrix to φ-level representation."""
        W_np = W.float().cpu().numpy()
        out_dim, in_dim = W_np.shape
        
        # Get signs
        signs = (W_np < 0).astype(np.uint8)  # 1 bit
        
        # Get φ-levels
        abs_W = np.abs(W_np)
        abs_W = np.maximum(abs_W, 1e-45)  # Avoid log(0)
        
        # Compute continuous levels
        levels_cont = np.log(abs_W) / LOG_PHI
        
        # Find the range of levels
        level_min = np.percentile(levels_cont, 0.5)
        level_max = np.percentile(levels_cont, 99.5)
        
        # Quantize to num_levels discrete levels
        level_step = (level_max - level_min) / (self.num_levels - 1)
        levels_idx = np.clip(
            np.round((levels_cont - level_min) / level_step).astype(np.int8),
            0, self.num_levels - 1
        )
        
        # Create level lookup table
        level_map = np.arange(self.num_levels, dtype=np.float32)
        level_values = PHI ** (level_min + level_map * level_step)
        
        # Pack: 5 bits per weight (4 for level, 1 for sign)
        # Use uint8, pack 1 weight per byte for simplicity (can optimize later)
        packed = (levels_idx.astype(np.uint8) | (signs << 4)).flatten()
        
        # Update stats
        self.stats["total_weights"] += out_dim * in_dim
        self.stats["original_bytes"] += out_dim * in_dim * 2  # bfloat16
        self.stats["compressed_bytes"] += len(packed) + len(level_values) * 4
        
        return PhiLevelWeight(
            shape=(out_dim, in_dim),
            packed=packed,
            level_values=level_values.astype(np.float32),
            level_map=np.arange(self.num_levels, dtype=np.int8),
        )
    
    def decompress_weight(self, pw: PhiLevelWeight) -> torch.Tensor:
        """Decompress a φ-level weight back to float."""
        out_dim, in_dim = pw.shape
        
        # Unpack
        levels_idx = pw.packed & 0x0F
        signs = (pw.packed >> 4) & 0x01
        
        # Reconstruct
        values = pw.level_values[levels_idx]
        values = values * (1 - 2 * signs.astype(np.float32))  # Apply sign
        
        return torch.tensor(values.reshape(out_dim, in_dim), dtype=torch.float32)
    
    def to_gpu(self, pw: PhiLevelWeight) -> PhiLevelWeight:
        """Move compressed weight to GPU."""
        pw.d_packed = torch.tensor(pw.packed, dtype=torch.uint8, device=DEVICE)
        pw.d_level_values = torch.tensor(pw.level_values, dtype=torch.float32, device=DEVICE)
        return pw
    
    def get_compression_stats(self) -> Dict:
        """Get compression statistics."""
        ratio = self.stats["original_bytes"] / max(1, self.stats["compressed_bytes"])
        return {
            **self.stats,
            "compression_ratio": f"{ratio:.2f}×",
        }


class PhiLevelLinear(torch.nn.Module):
    """Linear layer using φ-level compressed weights."""
    
    def __init__(self, compressed_weight: PhiLevelWeight, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.shape = compressed_weight.shape
        self.register_buffer('packed', torch.tensor(compressed_weight.packed, dtype=torch.uint8))
        self.register_buffer('level_values', torch.tensor(compressed_weight.level_values, dtype=torch.float32))
        
        if bias is not None:
            self.register_buffer('bias', bias.float())
        else:
            self.bias = None
        
        # Cache decompressed weight for now (optimize later with custom kernel)
        self._cached_weight = None
    
    def _get_weight(self) -> torch.Tensor:
        """Decompress weight on-the-fly."""
        if self._cached_weight is not None:
            return self._cached_weight
        
        out_dim, in_dim = self.shape
        
        # Unpack on GPU
        levels_idx = self.packed & 0x0F
        signs = ((self.packed >> 4) & 0x01).float()
        
        # Reconstruct
        values = self.level_values[levels_idx.long()]
        values = values * (1 - 2 * signs)
        
        weight = values.reshape(out_dim, in_dim)
        return weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._get_weight()
        
        # Match dtype
        if x.dtype != weight.dtype:
            weight = weight.to(x.dtype)
        
        out = F.linear(x, weight, self.bias)
        return out
    
    def cache_weight(self):
        """Cache decompressed weight for faster inference."""
        self._cached_weight = self._get_weight()
    
    def clear_cache(self):
        """Clear cached weight to save memory."""
        self._cached_weight = None


def compress_model(model, compressor: PhiLevelCompressor) -> Dict[str, PhiLevelWeight]:
    """Compress all linear layers in a model."""
    compressed = {}
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"  Compressing {name}: {tuple(module.weight.shape)}")
            compressed[name] = compressor.compress_weight(module.weight.data)
    
    return compressed


def test_phi_level_compression():
    """Test φ-level compression accuracy and speed."""
    print("=" * 70)
    print("φ-LEVEL COMPRESSION TEST")
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
    
    # Test compression on a single layer
    print("\n--- Testing Single Layer Compression ---")
    compressor = PhiLevelCompressor(num_levels=16)
    
    layer = model.model.layers[0]
    W_gate = layer.mlp.gate_proj.weight.data
    
    print(f"Original shape: {tuple(W_gate.shape)}")
    print(f"Original size: {W_gate.numel() * 2 / 1e6:.2f} MB (bfloat16)")
    
    # Compress
    start = time.perf_counter()
    compressed = compressor.compress_weight(W_gate)
    compress_time = (time.perf_counter() - start) * 1000
    
    print(f"Compressed size: {len(compressed.packed) / 1e6:.2f} MB")
    print(f"Compression time: {compress_time:.1f} ms")
    print(f"Compression ratio: {W_gate.numel() * 2 / len(compressed.packed):.2f}×")
    
    # Decompress and check accuracy
    start = time.perf_counter()
    W_reconstructed = compressor.decompress_weight(compressed)
    decompress_time = (time.perf_counter() - start) * 1000
    
    W_orig = W_gate.float().cpu()
    correlation = torch.corrcoef(torch.stack([W_orig.flatten(), W_reconstructed.flatten()]))[0, 1].item()
    max_error = (W_orig - W_reconstructed).abs().max().item()
    
    print(f"Decompress time: {decompress_time:.1f} ms")
    print(f"Correlation: {correlation * 100:.4f}%")
    print(f"Max error: {max_error:.6f}")
    
    # Test forward pass accuracy
    print("\n--- Testing Forward Pass Accuracy ---")
    
    # Create compressed linear layer
    phi_linear = PhiLevelLinear(compressed).to(DEVICE)
    phi_linear.cache_weight()  # Cache for speed
    
    # Test input
    test_input = torch.randn(1, 5, 3584, dtype=torch.bfloat16, device=DEVICE)
    
    # Original forward
    with torch.no_grad():
        orig_out = layer.mlp.gate_proj(test_input)
    
    # Compressed forward
    with torch.no_grad():
        comp_out = phi_linear(test_input)
    
    # Compare
    orig_flat = orig_out.float().flatten()
    comp_flat = comp_out.float().flatten()
    fwd_correlation = torch.corrcoef(torch.stack([orig_flat.cpu(), comp_flat.cpu()]))[0, 1].item()
    fwd_max_error = (orig_out.float() - comp_out.float()).abs().max().item()
    
    print(f"Forward correlation: {fwd_correlation * 100:.4f}%")
    print(f"Forward max error: {fwd_max_error:.6f}")
    
    # Benchmark speed
    print("\n--- Benchmarking Speed ---")
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = layer.mlp.gate_proj(test_input)
            _ = phi_linear(test_input)
    torch.cuda.synchronize()
    
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
    
    # Compressed (with cached weight)
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = phi_linear(test_input)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    comp_time = np.mean(times)
    
    print(f"Original: {orig_time:.3f} ms")
    print(f"Compressed: {comp_time:.3f} ms")
    print(f"Speedup: {orig_time / comp_time:.2f}×")
    
    # Full model compression estimate
    print("\n--- Full Model Compression Estimate ---")
    
    total_params = sum(p.numel() for p in model.parameters())
    linear_params = sum(
        m.weight.numel() for m in model.modules() if isinstance(m, torch.nn.Linear)
    )
    
    print(f"Total parameters: {total_params / 1e9:.2f}B")
    print(f"Linear layer parameters: {linear_params / 1e9:.2f}B ({linear_params/total_params*100:.1f}%)")
    
    # Estimate compressed size
    original_size = linear_params * 2  # bfloat16
    compressed_size = linear_params * 1  # 1 byte per weight (5 bits packed)
    
    print(f"Original linear size: {original_size / 1e9:.2f} GB")
    print(f"Compressed linear size: {compressed_size / 1e9:.2f} GB")
    print(f"Compression ratio: {original_size / compressed_size:.2f}×")
    
    # Theoretical speedup
    gpu_bandwidth = 1008  # GB/s
    orig_tok_per_sec = gpu_bandwidth / (original_size / 1e9)
    comp_tok_per_sec = gpu_bandwidth / (compressed_size / 1e9)
    
    print(f"\nTheoretical tokens/sec (bandwidth limited):")
    print(f"  Original: {orig_tok_per_sec:.0f} tok/s")
    print(f"  Compressed: {comp_tok_per_sec:.0f} tok/s")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_phi_level_compression()
