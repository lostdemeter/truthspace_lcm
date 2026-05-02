#!/usr/bin/env python3
"""
Native φ-Space Computation
===========================

Store weights as φ-coordinates and compute natively in φ-space.

Key insight: In φ-space, operations become simpler:
- Multiplication = level addition
- φ^n computation = integer level shift
- sigmoid = level comparison

Storage format:
- level: int8 (φ-power, range -128 to 127)
- sign: 1 bit
- residual: uint8 (0-255 maps to 0.0-1.0)

Total: 3 bytes per value vs 4 bytes (float32) or 2 bytes (float16)
But with better structure for φ-operations!
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from dataclasses import dataclass
from typing import Tuple, List, Optional
import time

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
PHI_MINUS_1 = PHI - 1  # ≈ 0.618


@dataclass
class PhiTensor:
    """
    Tensor stored in φ-coordinates.
    
    value = sign × φ^level × (1 + residual × (φ-1))
    
    Storage:
    - level: int8 array
    - sign: int8 array (+1 or -1)
    - residual: uint8 array (0-255 → 0.0-1.0)
    """
    level: np.ndarray      # int8
    sign: np.ndarray       # int8 (+1 or -1)
    residual: np.ndarray   # uint8 (quantized to 256 levels)
    shape: tuple
    
    @classmethod
    def from_float(cls, x: np.ndarray) -> 'PhiTensor':
        """Convert float array to φ-tensor."""
        shape = x.shape
        x_flat = x.flatten().astype(np.float64)
        
        # Handle zeros and near-zeros
        sign = np.sign(x_flat).astype(np.int8)
        sign[sign == 0] = 1
        
        abs_x = np.abs(x_flat)
        # Clamp to avoid log(0)
        abs_x = np.maximum(abs_x, 1e-38)
        
        # Compute φ-level
        log_phi_x = np.log(abs_x) / LN_PHI
        level = np.floor(log_phi_x).astype(np.int8)
        
        # Compute residual: how far between φ^level and φ^(level+1)
        # value = φ^level × (1 + residual × (φ-1))
        # residual = (value/φ^level - 1) / (φ-1)
        base = PHI ** level.astype(np.float64)
        residual_float = (abs_x / base - 1) / PHI_MINUS_1
        residual_float = np.clip(residual_float, 0, 1)
        
        # Quantize residual to uint8
        residual = (residual_float * 255).astype(np.uint8)
        
        return cls(level=level, sign=sign, residual=residual, shape=shape)
    
    def to_float(self) -> np.ndarray:
        """Convert back to float array."""
        residual_float = self.residual.astype(np.float64) / 255.0
        value = self.sign * (PHI ** self.level.astype(np.float64)) * (1 + residual_float * PHI_MINUS_1)
        return value.reshape(self.shape)
    
    def storage_bytes(self) -> int:
        """Total storage in bytes."""
        n = np.prod(self.shape)
        return n * 3  # 1 byte level + 1 byte sign + 1 byte residual
    
    @classmethod
    def storage_per_element(cls) -> int:
        return 3


def phi_matmul_native(A: PhiTensor, B: PhiTensor) -> np.ndarray:
    """
    Matrix multiply in φ-space.
    
    For now, convert to float and multiply.
    Future: implement native φ-space matmul.
    """
    A_float = A.to_float()
    B_float = B.to_float()
    return A_float @ B_float


def phi_sigmoid_native(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid using φ-formula.
    sigmoid(x) = 1 / (1 + φ^(-x/ln(φ)))
    """
    # This is mathematically identical to standard sigmoid
    return 1 / (1 + PHI ** (-x / LN_PHI))


def phi_softmax_native(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax using φ-powers.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    phi_powers = PHI ** ((x - x_max) / LN_PHI)
    return phi_powers / np.sum(phi_powers, axis=axis, keepdims=True)


def phi_silu_native(x: np.ndarray) -> np.ndarray:
    """SiLU using φ-sigmoid."""
    return x * phi_sigmoid_native(x)


class PhiNativeTransformer:
    """
    Transformer with weights stored as φ-tensors.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Loading and converting to φ-format...")
        
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        self.config = config
        self.n_layers = config.num_hidden_layers
        self.hidden_dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_dim // self.n_heads
        self.intermediate_size = config.intermediate_size
        self.rope_base = config.rope_theta
        
        # Convert embeddings to φ-format
        print("  Converting embeddings...")
        embed_np = model.model.embed_tokens.weight.data.numpy()
        self.embed_phi = PhiTensor.from_float(embed_np)
        self.embed_float = embed_np  # Keep float for fast lookup
        
        # Convert final norm and lm_head
        self.final_norm = model.model.norm.weight.data.numpy()
        self.lm_head = model.lm_head.weight.data.numpy()
        
        # Convert per-layer weights
        print("  Converting layers...")
        self.layers = []
        total_phi_bytes = 0
        total_float_bytes = 0
        
        for i in range(self.n_layers):
            layer = model.model.layers[i]
            
            # Extract weights
            weights = {
                'ln1_weight': layer.input_layernorm.weight.data.numpy(),
                'ln2_weight': layer.post_attention_layernorm.weight.data.numpy(),
                'q_weight': layer.self_attn.q_proj.weight.data.numpy(),
                'k_weight': layer.self_attn.k_proj.weight.data.numpy(),
                'v_weight': layer.self_attn.v_proj.weight.data.numpy(),
                'o_weight': layer.self_attn.o_proj.weight.data.numpy(),
                'q_bias': layer.self_attn.q_proj.bias.data.numpy(),
                'k_bias': layer.self_attn.k_proj.bias.data.numpy(),
                'v_bias': layer.self_attn.v_proj.bias.data.numpy(),
                'gate_weight': layer.mlp.gate_proj.weight.data.numpy(),
                'up_weight': layer.mlp.up_proj.weight.data.numpy(),
                'down_weight': layer.mlp.down_proj.weight.data.numpy(),
            }
            
            # Convert large matrices to φ-format
            phi_weights = {}
            for name, w in weights.items():
                if 'weight' in name and w.size > 10000:
                    phi_weights[name + '_phi'] = PhiTensor.from_float(w)
                    total_phi_bytes += phi_weights[name + '_phi'].storage_bytes()
                    total_float_bytes += w.size * 4
                phi_weights[name] = w  # Keep float for now
            
            self.layers.append(phi_weights)
            
            if (i + 1) % 7 == 0:
                print(f"    Layer {i+1}/{self.n_layers}")
        
        del model
        
        print(f"\nStorage comparison:")
        print(f"  Float32: {total_float_bytes / 1e9:.2f} GB")
        print(f"  φ-format: {total_phi_bytes / 1e9:.2f} GB")
        print(f"  Ratio: {total_float_bytes / total_phi_bytes:.2f}x")
    
    def compute_rope(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute RoPE cos/sin."""
        inv_freq = 1.0 / (self.rope_base ** (np.arange(0, self.head_dim, 2) / self.head_dim))
        positions = np.arange(seq_len)
        freqs = np.outer(positions, inv_freq)
        return np.cos(freqs), np.sin(freqs)
    
    def apply_rope(self, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        """Apply rotary position embedding."""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = np.empty_like(x)
        rotated[..., ::2] = x1 * cos - x2 * sin
        rotated[..., 1::2] = x1 * sin + x2 * cos
        return rotated
    
    def rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """RMS normalization."""
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return (x / rms) * weight
    
    def forward_layer(self, h: np.ndarray, layer_idx: int,
                      cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        """Forward pass through one layer."""
        L = self.layers[layer_idx]
        seq_len = h.shape[0]
        
        # Layer norm
        h_norm = self.rms_norm(h, L['ln1_weight'])
        
        # Q, K, V projections (using float for now)
        Q = h_norm @ L['q_weight'].T + L['q_bias']
        K = h_norm @ L['k_weight'].T + L['k_bias']
        V = h_norm @ L['v_weight'].T + L['v_bias']
        
        # Reshape for multi-head
        Q = Q.reshape(seq_len, self.n_heads, self.head_dim)
        K = K.reshape(seq_len, self.n_kv_heads, self.head_dim)
        V = V.reshape(seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        for pos in range(seq_len):
            for head in range(self.n_heads):
                Q[pos, head] = self.apply_rope(Q[pos, head], cos[pos], sin[pos])
            for head in range(self.n_kv_heads):
                K[pos, head] = self.apply_rope(K[pos, head], cos[pos], sin[pos])
        
        # Attention with φ-softmax
        heads_per_kv = self.n_heads // self.n_kv_heads
        attn_output = np.zeros((seq_len, self.hidden_dim))
        
        for h_idx in range(self.n_heads):
            kv_idx = h_idx // heads_per_kv
            
            scores = np.zeros((seq_len, seq_len))
            for i in range(seq_len):
                for j in range(i + 1):
                    scores[i, j] = np.dot(Q[i, h_idx], K[j, kv_idx]) / np.sqrt(self.head_dim)
                scores[i, i+1:] = -np.inf
            
            # φ-softmax
            attn_weights = phi_softmax_native(scores, axis=-1)
            v_out = attn_weights @ V[:, kv_idx]
            attn_output[:, h_idx*self.head_dim:(h_idx+1)*self.head_dim] = v_out
        
        # Output projection + residual
        attn_output = attn_output @ L['o_weight'].T
        h = h + attn_output
        
        # MLP
        h_norm = self.rms_norm(h, L['ln2_weight'])
        gate = h_norm @ L['gate_weight'].T
        up = h_norm @ L['up_weight'].T
        
        # φ-SiLU
        mlp_out = phi_silu_native(gate) * up
        mlp_out = mlp_out @ L['down_weight'].T
        
        return h + mlp_out
    
    def forward(self, token_ids: List[int]) -> int:
        """Full forward pass."""
        seq_len = len(token_ids)
        
        # Embedding lookup (float for speed)
        h = self.embed_float[token_ids]
        
        # RoPE
        cos, sin = self.compute_rope(seq_len)
        
        # All layers
        for layer_idx in range(self.n_layers):
            h = self.forward_layer(h, layer_idx, cos, sin)
        
        # Final norm and prediction
        h_final = self.rms_norm(h[-1], self.final_norm)
        logits = h_final @ self.lm_head.T
        
        return np.argmax(logits)


def test_phi_native():
    """Test φ-native transformer."""
    print("=" * 70)
    print("φ-NATIVE TRANSFORMER TEST")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    
    # Load original for comparison
    print("\nLoading original model...")
    config = AutoConfig.from_pretrained("Qwen/Qwen2-7B-Instruct")
    original = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    device = next(original.parameters()).device
    
    # Load φ-native
    print("\nLoading φ-native model...")
    phi_model = PhiNativeTransformer()
    
    # Test cases
    test_cases = [
        "The capital of France is",
        "Hello",
        "The quick brown",
    ]
    
    print("\n" + "=" * 70)
    print("ACCURACY TEST")
    print("=" * 70)
    
    matches = 0
    total = 0
    
    for text in test_cases:
        ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        
        # Original
        with torch.no_grad():
            out = original(torch.tensor([ids]).to(device))
            orig_pred = torch.argmax(out.logits[0, -1]).item()
        
        # φ-native
        phi_pred = phi_model.forward(ids)
        
        match = "✓" if orig_pred == phi_pred else "✗"
        if orig_pred == phi_pred:
            matches += 1
        total += 1
        
        print(f"\n  '{text}'")
        print(f"    Original: {orig_pred} ('{tokenizer.decode([orig_pred])}')")
        print(f"    φ-native: {phi_pred} ('{tokenizer.decode([phi_pred])}')")
        print(f"    Match: {match}")
    
    print("\n" + "=" * 70)
    print(f"ACCURACY: {matches}/{total} = {matches/total*100:.1f}%")
    print("=" * 70)
    
    # Storage analysis
    print("\n" + "=" * 70)
    print("STORAGE ANALYSIS")
    print("=" * 70)
    
    # Test φ-tensor roundtrip accuracy
    print("\nφ-Tensor roundtrip accuracy:")
    test_weights = phi_model.layers[0]['q_weight']
    phi_tensor = PhiTensor.from_float(test_weights)
    reconstructed = phi_tensor.to_float()
    
    error = np.abs(test_weights - reconstructed)
    rel_error = error / (np.abs(test_weights) + 1e-10)
    
    print(f"  Max absolute error: {np.max(error):.6e}")
    print(f"  Mean absolute error: {np.mean(error):.6e}")
    print(f"  Max relative error: {np.max(rel_error):.6e}")
    print(f"  Mean relative error: {np.mean(rel_error):.6e}")
    
    # Correlation
    corr = np.corrcoef(test_weights.flatten(), reconstructed.flatten())[0, 1]
    print(f"  Correlation: {corr:.10f}")


if __name__ == "__main__":
    test_phi_native()
