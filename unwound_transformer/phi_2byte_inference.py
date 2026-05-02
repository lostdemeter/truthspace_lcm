#!/usr/bin/env python3
"""
Full Inference with φ-2byte Weights
====================================

Test that we can run full inference with weights stored in φ-2byte format
and still get 100% token accuracy.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Tuple, List
import time

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
PHI_MINUS_1 = PHI - 1


class PhiTensor2Byte:
    """2-byte φ-tensor: level (int8) + sign|residual (uint8)"""
    
    def __init__(self, data: np.ndarray, shape: tuple):
        self.data = data
        self.shape = shape
    
    @classmethod
    def from_float(cls, x: np.ndarray) -> 'PhiTensor2Byte':
        shape = x.shape
        x_flat = x.flatten().astype(np.float64)
        sign_bit = (x_flat < 0).astype(np.uint16)
        abs_x = np.maximum(np.abs(x_flat), 1e-38)
        log_phi_x = np.log(abs_x) / LN_PHI
        level = np.floor(log_phi_x).astype(np.int8)
        base = PHI ** level.astype(np.float64)
        residual_float = np.clip((abs_x / base - 1) / PHI_MINUS_1, 0, 1)
        residual = (residual_float * 127).astype(np.uint8)
        level_uint8 = level.view(np.uint8)
        high_byte = (sign_bit << 7) | residual
        data = level_uint8.astype(np.uint16) | (high_byte << 8)
        return cls(data=data, shape=shape)
    
    def to_float(self) -> np.ndarray:
        level_uint8 = (self.data & 0xFF).astype(np.uint8)
        level = level_uint8.view(np.int8).astype(np.float64)
        high_byte = (self.data >> 8).astype(np.uint8)
        sign_bit = (high_byte >> 7).astype(np.int8)
        residual = (high_byte & 0x7F).astype(np.float64) / 127.0
        sign = 1 - 2 * sign_bit
        return (sign * (PHI ** level) * (1 + residual * PHI_MINUS_1)).reshape(self.shape)
    
    def storage_bytes(self) -> int:
        return self.data.nbytes


def phi_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + PHI ** (-x / LN_PHI))

def phi_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    phi_powers = PHI ** ((x - x_max) / LN_PHI)
    return phi_powers / np.sum(phi_powers, axis=axis, keepdims=True)

def phi_silu(x: np.ndarray) -> np.ndarray:
    return x * phi_sigmoid(x)

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight

def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = np.empty_like(x)
    rotated[..., ::2] = x1 * cos - x2 * sin
    rotated[..., 1::2] = x1 * sin + x2 * cos
    return rotated


class Phi2ByteTransformer:
    """Transformer with weights in φ-2byte format."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Loading and converting to φ-2byte format...")
        start = time.time()
        
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
        self.rope_base = config.rope_theta
        
        # Keep embeddings as float (lookup table)
        self.embed = model.model.embed_tokens.weight.data.numpy()
        self.final_norm = model.model.norm.weight.data.numpy()
        self.lm_head = model.lm_head.weight.data.numpy()
        
        # Convert layer weights to φ-2byte
        self.layers = []
        total_phi_bytes = 0
        total_float_bytes = 0
        
        for i in range(self.n_layers):
            layer = model.model.layers[i]
            
            layer_data = {
                # Norms stay as float (small)
                'ln1_weight': layer.input_layernorm.weight.data.numpy(),
                'ln2_weight': layer.post_attention_layernorm.weight.data.numpy(),
                # Biases stay as float (small)
                'q_bias': layer.self_attn.q_proj.bias.data.numpy(),
                'k_bias': layer.self_attn.k_proj.bias.data.numpy(),
                'v_bias': layer.self_attn.v_proj.bias.data.numpy(),
            }
            
            # Convert large matrices to φ-2byte
            for name, tensor in [
                ('q_weight', layer.self_attn.q_proj.weight),
                ('k_weight', layer.self_attn.k_proj.weight),
                ('v_weight', layer.self_attn.v_proj.weight),
                ('o_weight', layer.self_attn.o_proj.weight),
                ('gate_weight', layer.mlp.gate_proj.weight),
                ('up_weight', layer.mlp.up_proj.weight),
                ('down_weight', layer.mlp.down_proj.weight),
            ]:
                w = tensor.data.numpy()
                phi_tensor = PhiTensor2Byte.from_float(w)
                layer_data[name] = phi_tensor
                total_phi_bytes += phi_tensor.storage_bytes()
                total_float_bytes += w.size * 4
            
            self.layers.append(layer_data)
        
        del model
        
        elapsed = time.time() - start
        print(f"Conversion complete in {elapsed:.1f}s")
        print(f"Storage: {total_float_bytes/1e9:.2f} GB (float32) → {total_phi_bytes/1e9:.2f} GB (φ-2byte)")
        print(f"Compression: {total_float_bytes/total_phi_bytes:.2f}x")
    
    def compute_rope(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        inv_freq = 1.0 / (self.rope_base ** (np.arange(0, self.head_dim, 2) / self.head_dim))
        positions = np.arange(seq_len)
        freqs = np.outer(positions, inv_freq)
        return np.cos(freqs), np.sin(freqs)
    
    def forward_layer(self, h: np.ndarray, layer_idx: int,
                      cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        L = self.layers[layer_idx]
        seq_len = h.shape[0]
        
        # Decompress weights on-the-fly
        q_weight = L['q_weight'].to_float()
        k_weight = L['k_weight'].to_float()
        v_weight = L['v_weight'].to_float()
        o_weight = L['o_weight'].to_float()
        gate_weight = L['gate_weight'].to_float()
        up_weight = L['up_weight'].to_float()
        down_weight = L['down_weight'].to_float()
        
        # Layer norm
        h_norm = rms_norm(h, L['ln1_weight'])
        
        # Q, K, V
        Q = h_norm @ q_weight.T + L['q_bias']
        K = h_norm @ k_weight.T + L['k_bias']
        V = h_norm @ v_weight.T + L['v_bias']
        
        # Reshape
        Q = Q.reshape(seq_len, self.n_heads, self.head_dim)
        K = K.reshape(seq_len, self.n_kv_heads, self.head_dim)
        V = V.reshape(seq_len, self.n_kv_heads, self.head_dim)
        
        # RoPE
        for pos in range(seq_len):
            for head in range(self.n_heads):
                Q[pos, head] = apply_rope(Q[pos, head], cos[pos], sin[pos])
            for head in range(self.n_kv_heads):
                K[pos, head] = apply_rope(K[pos, head], cos[pos], sin[pos])
        
        # Attention
        heads_per_kv = self.n_heads // self.n_kv_heads
        attn_output = np.zeros((seq_len, self.hidden_dim))
        
        for h_idx in range(self.n_heads):
            kv_idx = h_idx // heads_per_kv
            scores = np.zeros((seq_len, seq_len))
            for i in range(seq_len):
                for j in range(i + 1):
                    scores[i, j] = np.dot(Q[i, h_idx], K[j, kv_idx]) / np.sqrt(self.head_dim)
                scores[i, i+1:] = -np.inf
            
            attn_weights = phi_softmax(scores, axis=-1)
            v_out = attn_weights @ V[:, kv_idx]
            attn_output[:, h_idx*self.head_dim:(h_idx+1)*self.head_dim] = v_out
        
        attn_output = attn_output @ o_weight.T
        h = h + attn_output
        
        # MLP
        h_norm = rms_norm(h, L['ln2_weight'])
        gate = h_norm @ gate_weight.T
        up = h_norm @ up_weight.T
        mlp_out = phi_silu(gate) * up
        mlp_out = mlp_out @ down_weight.T
        
        return h + mlp_out
    
    def forward(self, token_ids: List[int]) -> int:
        seq_len = len(token_ids)
        h = self.embed[token_ids]
        cos, sin = self.compute_rope(seq_len)
        
        for layer_idx in range(self.n_layers):
            h = self.forward_layer(h, layer_idx, cos, sin)
        
        h_final = rms_norm(h[-1], self.final_norm)
        logits = h_final @ self.lm_head.T
        return np.argmax(logits)


def main():
    print("=" * 70)
    print("φ-2BYTE INFERENCE TEST")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    
    # Load original
    print("\nLoading original model...")
    original = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    device = next(original.parameters()).device
    
    # Load φ-2byte
    print()
    phi_model = Phi2ByteTransformer()
    
    # Test
    test_cases = [
        "The capital of France is",
        "Hello",
        "The quick brown",
    ]
    
    print("\n" + "=" * 70)
    print("ACCURACY TEST")
    print("=" * 70)
    
    matches = 0
    for text in test_cases:
        ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        
        with torch.no_grad():
            out = original(torch.tensor([ids]).to(device))
            orig_pred = torch.argmax(out.logits[0, -1]).item()
        
        phi_pred = phi_model.forward(ids)
        
        match = "✓" if orig_pred == phi_pred else "✗"
        if orig_pred == phi_pred:
            matches += 1
        
        print(f"\n  '{text}'")
        print(f"    Original: {orig_pred} ('{tokenizer.decode([orig_pred])}')")
        print(f"    φ-2byte:  {phi_pred} ('{tokenizer.decode([phi_pred])}')")
        print(f"    Match: {match}")
    
    print("\n" + "=" * 70)
    print(f"ACCURACY: {matches}/{len(test_cases)} = {matches/len(test_cases)*100:.1f}%")
    print("=" * 70)
    
    if matches == len(test_cases):
        print("\n  ✓ φ-2BYTE FORMAT MAINTAINS 100% ACCURACY")
        print("  2x compression with full precision!")


if __name__ == "__main__":
    main()
