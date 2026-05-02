#!/usr/bin/env python3
"""
Adaptive LOD with φ-Levels
===========================

Combine:
1. φ-2byte storage (2× compression)
2. Hierarchical φ-encoding (multi-level precision)
3. Adaptive LOD (use fewer levels for easy tokens)

Key insight: Weights cluster at specific φ-levels (φ^-9 peak).
Easy tokens only need the dominant levels.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Tuple, List, Dict
import time

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
PHI_MINUS_1 = PHI - 1


class PhiLODTensor:
    """
    φ-tensor with Level-of-Detail support.
    
    Stores weights decomposed by φ-level, allowing partial reconstruction
    using only the most important levels.
    """
    
    def __init__(self, level_data: Dict[int, np.ndarray], shape: tuple, 
                 level_order: List[int]):
        """
        level_data: {level: (indices, signs, residuals)} for each φ-level
        level_order: levels sorted by importance (most values first)
        """
        self.level_data = level_data
        self.shape = shape
        self.level_order = level_order
        self.n_elements = np.prod(shape)
    
    @classmethod
    def from_float(cls, x: np.ndarray) -> 'PhiLODTensor':
        """Convert float array to LOD φ-tensor."""
        shape = x.shape
        x_flat = x.flatten().astype(np.float64)
        n = len(x_flat)
        
        # Compute φ-levels for all values
        sign = np.sign(x_flat)
        sign[sign == 0] = 1
        
        abs_x = np.maximum(np.abs(x_flat), 1e-38)
        log_phi_x = np.log(abs_x) / LN_PHI
        levels = np.floor(log_phi_x).astype(np.int8)
        
        # Compute residuals
        base = PHI ** levels.astype(np.float64)
        residual_float = np.clip((abs_x / base - 1) / PHI_MINUS_1, 0, 1)
        residuals = (residual_float * 127).astype(np.uint8)
        
        # Group by level
        unique_levels = np.unique(levels)
        level_data = {}
        level_counts = []
        
        for lvl in unique_levels:
            mask = levels == lvl
            indices = np.where(mask)[0].astype(np.uint32)
            level_data[int(lvl)] = {
                'indices': indices,
                'signs': sign[mask].astype(np.int8),
                'residuals': residuals[mask],
            }
            level_counts.append((int(lvl), len(indices)))
        
        # Sort levels by count (most common first)
        level_counts.sort(key=lambda x: -x[1])
        level_order = [lvl for lvl, _ in level_counts]
        
        return cls(level_data=level_data, shape=shape, level_order=level_order)
    
    def to_float_full(self) -> np.ndarray:
        """Reconstruct full precision."""
        return self.to_float_lod(len(self.level_order))
    
    def to_float_lod(self, n_levels: int) -> np.ndarray:
        """Reconstruct using only top n_levels."""
        result = np.zeros(self.n_elements, dtype=np.float64)
        
        for i, lvl in enumerate(self.level_order[:n_levels]):
            if lvl not in self.level_data:
                continue
            
            data = self.level_data[lvl]
            indices = data['indices']
            signs = data['signs'].astype(np.float64)
            residuals = data['residuals'].astype(np.float64) / 127.0
            
            values = signs * (PHI ** lvl) * (1 + residuals * PHI_MINUS_1)
            result[indices] = values
        
        return result.reshape(self.shape)
    
    def coverage_at_lod(self, n_levels: int) -> float:
        """What fraction of values are covered by top n_levels?"""
        covered = sum(
            len(self.level_data[lvl]['indices']) 
            for lvl in self.level_order[:n_levels]
            if lvl in self.level_data
        )
        return covered / self.n_elements


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


class PhiAdaptiveLODTransformer:
    """
    Transformer with adaptive LOD based on φ-levels.
    
    - Easy tokens: use top 3 φ-levels (covers ~60% of weights)
    - Medium tokens: use top 6 φ-levels (covers ~90% of weights)
    - Hard tokens: use all levels (100%)
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print("Loading and converting to φ-LOD format...")
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
        
        # Keep embeddings as float
        self.embed = model.model.embed_tokens.weight.data.numpy()
        self.final_norm = model.model.norm.weight.data.numpy()
        self.lm_head = model.lm_head.weight.data.numpy()
        
        # Convert layer weights to φ-LOD format
        self.layers = []
        
        for i in range(self.n_layers):
            layer = model.model.layers[i]
            
            layer_data = {
                'ln1_weight': layer.input_layernorm.weight.data.numpy(),
                'ln2_weight': layer.post_attention_layernorm.weight.data.numpy(),
                'q_bias': layer.self_attn.q_proj.bias.data.numpy(),
                'k_bias': layer.self_attn.k_proj.bias.data.numpy(),
                'v_bias': layer.self_attn.v_proj.bias.data.numpy(),
            }
            
            # Convert large matrices to φ-LOD
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
                layer_data[name] = PhiLODTensor.from_float(w)
            
            self.layers.append(layer_data)
            
            if (i + 1) % 7 == 0:
                print(f"  Layer {i+1}/{self.n_layers}")
        
        del model
        
        # Analyze LOD coverage
        sample_layer = self.layers[0]['q_weight']
        print(f"\nLOD coverage analysis (layer 0 Q):")
        for n in [1, 2, 3, 4, 5, 6]:
            cov = sample_layer.coverage_at_lod(n)
            print(f"  Top {n} levels: {cov*100:.1f}% coverage")
        
        elapsed = time.time() - start
        print(f"\nConversion complete in {elapsed:.1f}s")
    
    def compute_rope(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        inv_freq = 1.0 / (self.rope_base ** (np.arange(0, self.head_dim, 2) / self.head_dim))
        positions = np.arange(seq_len)
        freqs = np.outer(positions, inv_freq)
        return np.cos(freqs), np.sin(freqs)
    
    def forward_layer_lod(self, h: np.ndarray, layer_idx: int,
                          cos: np.ndarray, sin: np.ndarray,
                          n_levels: int) -> np.ndarray:
        """Forward pass with LOD control."""
        L = self.layers[layer_idx]
        seq_len = h.shape[0]
        
        # Decompress weights at specified LOD
        q_weight = L['q_weight'].to_float_lod(n_levels)
        k_weight = L['k_weight'].to_float_lod(n_levels)
        v_weight = L['v_weight'].to_float_lod(n_levels)
        o_weight = L['o_weight'].to_float_lod(n_levels)
        gate_weight = L['gate_weight'].to_float_lod(n_levels)
        up_weight = L['up_weight'].to_float_lod(n_levels)
        down_weight = L['down_weight'].to_float_lod(n_levels)
        
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
    
    def forward_adaptive(self, token_ids: List[int], 
                         confidence_threshold: float = 0.9) -> Tuple[int, dict]:
        """
        Adaptive forward pass:
        1. Try with low LOD (3 levels)
        2. If confident, return
        3. Otherwise, refine with more levels
        """
        seq_len = len(token_ids)
        h = self.embed[token_ids]
        cos, sin = self.compute_rope(seq_len)
        
        stats = {'lod_used': [], 'confidences': []}
        
        # Start with low LOD
        n_levels = 3
        
        for layer_idx in range(self.n_layers):
            h = self.forward_layer_lod(h, layer_idx, cos, sin, n_levels)
        
        # Check confidence
        h_final = rms_norm(h[-1], self.final_norm)
        logits = h_final @ self.lm_head.T
        probs = phi_softmax(logits)
        confidence = np.max(probs)
        
        stats['confidences'].append(confidence)
        stats['lod_used'].append(n_levels)
        
        if confidence >= confidence_threshold:
            return np.argmax(logits), stats
        
        # Medium LOD
        n_levels = 6
        h = self.embed[token_ids]
        
        for layer_idx in range(self.n_layers):
            h = self.forward_layer_lod(h, layer_idx, cos, sin, n_levels)
        
        h_final = rms_norm(h[-1], self.final_norm)
        logits = h_final @ self.lm_head.T
        probs = phi_softmax(logits)
        confidence = np.max(probs)
        
        stats['confidences'].append(confidence)
        stats['lod_used'].append(n_levels)
        
        if confidence >= 0.5:
            return np.argmax(logits), stats
        
        # Full LOD
        n_levels = 20  # All levels
        h = self.embed[token_ids]
        
        for layer_idx in range(self.n_layers):
            h = self.forward_layer_lod(h, layer_idx, cos, sin, n_levels)
        
        h_final = rms_norm(h[-1], self.final_norm)
        logits = h_final @ self.lm_head.T
        
        stats['lod_used'].append(n_levels)
        
        return np.argmax(logits), stats
    
    def forward_full(self, token_ids: List[int]) -> int:
        """Full precision forward pass."""
        seq_len = len(token_ids)
        h = self.embed[token_ids]
        cos, sin = self.compute_rope(seq_len)
        
        for layer_idx in range(self.n_layers):
            h = self.forward_layer_lod(h, layer_idx, cos, sin, n_levels=20)
        
        h_final = rms_norm(h[-1], self.final_norm)
        logits = h_final @ self.lm_head.T
        return np.argmax(logits)


def main():
    print("=" * 70)
    print("ADAPTIVE LOD φ-TRANSFORMER TEST")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    
    # Load original for comparison
    print("\nLoading original model...")
    original = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    device = next(original.parameters()).device
    
    # Load φ-LOD model
    print()
    phi_model = PhiAdaptiveLODTransformer()
    
    # Test cases
    test_cases = [
        "The capital of France is",
        "Hello",
        "The quick brown",
        "Machine learning is",
        "In the year 2025",
    ]
    
    print("\n" + "=" * 70)
    print("ACCURACY TEST: FULL LOD vs ORIGINAL")
    print("=" * 70)
    
    matches_full = 0
    for text in test_cases:
        ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        
        with torch.no_grad():
            out = original(torch.tensor([ids]).to(device))
            orig_pred = torch.argmax(out.logits[0, -1]).item()
        
        phi_pred = phi_model.forward_full(ids)
        
        match = "✓" if orig_pred == phi_pred else "✗"
        if orig_pred == phi_pred:
            matches_full += 1
        
        print(f"  '{text}': {match}")
    
    print(f"\nFull LOD accuracy: {matches_full}/{len(test_cases)}")
    
    print("\n" + "=" * 70)
    print("ADAPTIVE LOD TEST")
    print("=" * 70)
    
    matches_adaptive = 0
    for text in test_cases:
        ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        
        with torch.no_grad():
            out = original(torch.tensor([ids]).to(device))
            orig_pred = torch.argmax(out.logits[0, -1]).item()
        
        phi_pred, stats = phi_model.forward_adaptive(ids)
        
        match = "✓" if orig_pred == phi_pred else "✗"
        if orig_pred == phi_pred:
            matches_adaptive += 1
        
        print(f"\n  '{text}'")
        print(f"    Original: {orig_pred} ('{tokenizer.decode([orig_pred])}')")
        print(f"    Adaptive: {phi_pred} ('{tokenizer.decode([phi_pred])}')")
        print(f"    LOD used: {stats['lod_used']}")
        print(f"    Confidences: {[f'{c:.3f}' for c in stats['confidences']]}")
        print(f"    Match: {match}")
    
    print("\n" + "=" * 70)
    print(f"ADAPTIVE LOD ACCURACY: {matches_adaptive}/{len(test_cases)}")
    print("=" * 70)
    
    # Timing comparison
    print("\n" + "=" * 70)
    print("TIMING COMPARISON")
    print("=" * 70)
    
    test_text = "The capital of France is"
    ids = tokenizer(test_text, return_tensors="pt").input_ids[0].tolist()
    
    # Time full LOD
    start = time.time()
    for _ in range(3):
        phi_model.forward_full(ids)
    full_time = (time.time() - start) / 3
    
    # Time low LOD (3 levels)
    start = time.time()
    for _ in range(3):
        h = phi_model.embed[ids]
        cos, sin = phi_model.compute_rope(len(ids))
        for layer_idx in range(phi_model.n_layers):
            h = phi_model.forward_layer_lod(h, layer_idx, cos, sin, n_levels=3)
    low_time = (time.time() - start) / 3
    
    print(f"\n  Full LOD (20 levels): {full_time:.2f}s")
    print(f"  Low LOD (3 levels):   {low_time:.2f}s")
    print(f"  Speedup: {full_time/low_time:.2f}x")


if __name__ == "__main__":
    main()
